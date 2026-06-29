"""
evaluate.py
───────────
ЗАПУСК:
    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/val
"""

import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from data.dataset import WagonDataset, get_val_transform
from models.model import WagonNumberOCR


def evaluate(checkpoint_path: str, data_dir: str = "data/val", batch_size: int = 64):
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    cfg    = ckpt.get("config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WagonNumberOCR(
        num_digits=cfg.get("num_digits", 7),
        num_classes=cfg.get("num_classes", 10),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds     = WagonDataset(data_dir, transform=get_val_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(-1).cpu()
            all_preds.append(preds)
            all_targets.append(labels)

    P = torch.cat(all_preds)    # (N, 7)
    T = torch.cat(all_targets)  # (N, 7)
    N = len(P)

    digit_acc = (P == T).float().mean().item()
    seq_acc   = (P == T).all(dim=1).float().mean().item()

    print("\n" + "═" * 52)
    print(f"  Samples: {N}   |   Checkpoint: {checkpoint_path}")
    print("═" * 52)
    print(f"  Digit accuracy    : {digit_acc*100:6.2f}%")
    print(f"  Sequence accuracy : {seq_acc*100:6.2f}%")
    print(f"  Sequence errors   : {int((1-seq_acc)*N)}")

    print("\n  Per-position accuracy:")
    for i in range(7):
        acc = (P[:, i] == T[:, i]).float().mean().item()
        bar = "█" * int(acc * 25)
        print(f"    pos {i+1}: {acc*100:5.1f}%  {bar}")

    errors = Counter()
    for p, t in zip(P.view(-1).tolist(), T.view(-1).tolist()):
        if p != t:
            errors[(t, p)] += 1

    print("\n  Top-10 confused pairs (true→pred):")
    for (t, p), cnt in errors.most_common(10):
        print(f"    {t}→{p}  :  {cnt}×")

    wrong = ~(P == T).all(dim=1)
    Pw, Tw = P[wrong], T[wrong]
    print(f"\n  First 10 sequence errors:")
    for i in range(min(10, len(Pw))):
        t_s = "".join(str(x) for x in Tw[i].tolist())
        p_s = "".join(str(x) for x in Pw[i].tolist())
        pos = [j+1 for j in range(7) if Tw[i][j] != Pw[i][j]]
        print(f"    true={t_s}  pred={p_s}  wrong_pos={pos}")

    print("═" * 52 + "\n")
    return {"digit_accuracy": digit_acc, "sequence_accuracy": seq_acc}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",   default="data/val")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.batch_size)
