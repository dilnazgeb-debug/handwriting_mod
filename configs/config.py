"""configs/config.py"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Пути
    train_dir:       str = "data/train"
    val_dir:         str = "data/val"
    checkpoint_dir:  str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pt"

    # Модель
    num_digits:  int   = 7
    num_classes: int   = 10
    dropout:     float = 0.4

    # Обучение
    epochs:        int   = 80
    freeze_epochs: int   = 20    # сначала учим только головы
    batch_size:    int   = 16    # маленький датасет → маленький batch
    num_workers:   int   = 2
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    grad_clip:     float = 1.0

    # Loss
    label_smoothing: float = 0.05
    ordinal_weight:  float = 0.05

    # Misc
    seed:            int  = 42
    device:          str  = "cuda"
    mixed_precision: bool = True
    log_every:       int  = 10

    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
