"""
train.py
────────
Двухфазное обучение:
  Фаза 1 (эпохи 1..freeze_epochs): backbone заморожен, учим только головы
  Фаза 2 (остальные эпохи):        размораживаем, fine-tune всего с маленьким lr

ЗАПУСК:
    python train.py
    python train.py --epochs 80 --batch_size 16
"""

import argparse
import random
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import WagonDataset, get_train_transform, get_val_transform
from models.model import WagonNumberOCR
from utils.loss import WagonOCRLoss
from utils.metrics import compute_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, lr, weight_decay, phase2=False):
    if not phase2:
        # Фаза 1: только головы + neck
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        # Фаза 2: backbone с lr/10, остальное с lr
        backbone_ids = set(id(p) for p in model.features.parameters())
        params = [
            {"params": [p for p in model.parameters() if id(p) in backbone_ids],
             "lr": lr / 10},
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
             "lr": lr},
        ]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, cfg, train=True):
    model.train() if train else model.eval()
    totals = {"loss": 0.0, "digit_acc": 0.0, "seq_acc": 0.0}
    ctx = autocast(device_type=device.type, enabled=cfg.mixed_precision)

    for step, (imgs, labels) in enumerate(loader, 1):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with ctx:
            logits = model(imgs)
            loss, loss_dict = criterion(logits, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            m = compute_metrics(logits, labels)
        totals["loss"]      += loss_dict["loss_total"]
        totals["digit_acc"] += m["digit_accuracy"]
        totals["seq_acc"]   += m["sequence_accuracy"]

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def main(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"\n🚀  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = WagonDataset(cfg.train_dir, transform=get_train_transform())
    val_ds   = WagonDataset(cfg.val_dir,   transform=get_val_transform())
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Используем WeightedSampler для балансировки
    sampler = train_ds.get_sampler()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = WagonNumberOCR(cfg.num_digits, cfg.num_classes, cfg.dropout).to(device)
    criterion = WagonOCRLoss(cfg.label_smoothing, cfg.ordinal_weight)
    scaler    = GradScaler(device=str(device), enabled=cfg.mixed_precision)

    best_seq_acc = 0.0
    freeze_epochs = cfg.freeze_epochs  # эпохи с замороженным backbone

    # ── Фаза 1: backbone заморожен ────────────────────────────────────────────
    print(f"\n── Фаза 1: backbone заморожен ({freeze_epochs} эпох) ──")
    model.freeze_backbone()
    optimizer = make_optimizer(model, cfg.lr, cfg.weight_decay, phase2=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        steps_per_epoch=len(train_loader),
        epochs=freeze_epochs, pct_start=0.2,
    )

    for epoch in range(1, freeze_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, cfg, train=True)
        va = run_epoch(model, val_loader,   None,      None,      criterion, scaler, device, cfg, train=False)
        _log(epoch, freeze_epochs, tr, va, best_seq_acc, cfg, model, "P1")
        best_seq_acc = _save_best(model, va, best_seq_acc, epoch, cfg)

    # ── Фаза 2: fine-tune всего ───────────────────────────────────────────────
    remaining = cfg.epochs - freeze_epochs
    print(f"\n── Фаза 2: fine-tune всего ({remaining} эпох, backbone lr={cfg.lr/10:.1e}) ──")
    model.unfreeze_backbone()
    optimizer = make_optimizer(model, cfg.lr * 0.3, cfg.weight_decay, phase2=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining * len(train_loader), eta_min=1e-6,
    )

    for epoch in range(freeze_epochs + 1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, cfg, train=True)
        va = run_epoch(model, val_loader,   None,      None,      criterion, scaler, device, cfg, train=False)
        _log(epoch, cfg.epochs, tr, va, best_seq_acc, cfg, model, "P2")
        best_seq_acc = _save_best(model, va, best_seq_acc, epoch, cfg)

    print(f"\n✅  Done. Best val seq_acc: {best_seq_acc:.4f}")


def _log(epoch, total, tr, va, best, cfg, model, phase):
    flag = "  ← ✓ BEST" if va["seq_acc"] > best else ""
    print(f"[{phase}] E{epoch:03d}/{total:03d}  "
          f"TRAIN digit={tr['digit_acc']:.3f} seq={tr['seq_acc']:.3f}  |  "
          f"VAL digit={va['digit_acc']:.3f} seq={va['seq_acc']:.3f}{flag}")


def _save_best(model, va, best, epoch, cfg):
    if va["seq_acc"] > best:
        best = va["seq_acc"]
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_seq_acc": best,
            "config": cfg.__dict__,
        }, cfg.best_model_path)
        print(f"  💾  Saved → {cfg.best_model_path}  (seq_acc={best:.4f})")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--freeze_epochs", type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--device",        type=str,   default=None)
    args = parser.parse_args()

    cfg = Config()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    main(cfg)
