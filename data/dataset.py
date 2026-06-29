"""
data/dataset.py
"""

import re
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

IMG_H, IMG_W = 64, 256   # меньший размер — быстрее на малом датасете
IMG_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_RE = re.compile(r"^(\d{7})")


class WagonDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.transform = transform
        self.samples = []

        for fpath in sorted(Path(root).iterdir()):
            if fpath.suffix.lower() not in IMG_EXTS:
                continue
            m = LABEL_RE.match(fpath.stem)
            if m:
                label = [int(c) for c in m.group(1)]
                self.samples.append((fpath, label))

        if not self.samples:
            raise RuntimeError(
                f"Нет размеченных файлов в {root!r}.\n"
                "Имена файлов должны начинаться с 7 цифр: 1616408_xxx.jpg"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img_np = np.array(Image.open(fpath).convert("L"))

        if self.transform:
            img_t = self.transform(image=img_np)["image"]
        else:
            img_t = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0

        return img_t, torch.tensor(label, dtype=torch.long)

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Балансировка по первой цифре номера (самая важная позиция).
        Помогает при дисбалансе классов.
        """
        first_digits = [s[1][0] for s in self.samples]
        counts = Counter(first_digits)
        weights = [1.0 / counts[d] for d in first_digits]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_train_transform():
    return A.Compose([
        A.Resize(IMG_H, IMG_W),

        # Лёгкие геометрические (не рвать цифры)
        A.ShiftScaleRotate(
            shift_limit=0.04, scale_limit=0.08, rotate_limit=5,
            border_mode=0, value=255, p=0.6,
        ),
        A.Perspective(scale=(0.01, 0.04), p=0.3),

        # Качество скана
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=3),
        ], p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.ImageCompression(quality_lower=65, quality_upper=95, p=0.3),

        # Толщина линий ручки
        A.OneOf([
            A.Morphological(scale=(1, 2), operation="dilation"),
            A.Morphological(scale=(1, 2), operation="erosion"),
        ], p=0.35),

        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMG_H, IMG_W),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
