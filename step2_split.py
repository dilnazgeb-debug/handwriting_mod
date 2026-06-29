"""
step2_split.py
──────────────
Берёт все размеченные кропы из crops/
Разбивает на data/train/ и data/val/ (по умолчанию 85% / 15%)

ЗАПУСК:
    python step2_split.py
    python step2_split.py --val_ratio 0.2   # 20% на валидацию
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import Counter

CROP_DIR  = Path("crops")
TRAIN_DIR = Path("data/train")
VAL_DIR   = Path("data/val")
IMG_EXTS  = {".jpg", ".jpeg", ".png"}


def main(val_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)

    files = sorted([f for f in CROP_DIR.iterdir() if f.suffix.lower() in IMG_EXTS])
    if not files:
        print(f"❌  Нет файлов в {CROP_DIR}/. Сначала запусти step1_crop_and_label.py")
        return

    # Очищаем предыдущий сплит
    for d in [TRAIN_DIR, VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()

    random.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_files   = files[:n_val]
    train_files = files[n_val:]

    for f in train_files:
        shutil.copy(f, TRAIN_DIR / f.name)
    for f in val_files:
        shutil.copy(f, VAL_DIR / f.name)

    # Статистика по цифрам
    all_labels = [f.stem.split("_")[0] for f in files]
    digit_counts = Counter("".join(all_labels))

    print(f"\n✓ Сплит готов:")
    print(f"   Train : {len(train_files)} файлов → data/train/")
    print(f"   Val   : {len(val_files)} файлов  → data/val/")
    print(f"\nРаспределение цифр в датасете:")
    for digit in "0123456789":
        count = digit_counts.get(digit, 0)
        bar = "█" * (count // max(1, max(digit_counts.values()) // 20))
        print(f"   {digit}: {count:4d}  {bar}")

    print(f"\nСледующий шаг: python train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    main(args.val_ratio, args.seed)
