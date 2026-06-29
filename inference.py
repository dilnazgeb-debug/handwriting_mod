"""
inference.py
────────────
Дроп-ин замена EasyOCR для поля wagon_number.

ПРИМЕР ИСПОЛЬЗОВАНИЯ в твоём pipeline:
────────────────────────────────────────
from inference import WagonOCRInference

ocr = WagonOCRInference("checkpoints/best_model.pt")

# Вариант 1: просто предсказание
number = ocr.predict(crop_image)        # → "1616408"

# Вариант 2: с уверенностью (для фолбэка на EasyOCR)
number, conf, ok = ocr.predict_with_confidence(crop_image)
if not ok:
    number = easyocr_fallback(crop_image)
"""

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.model import WagonNumberOCR
from data.dataset import IMG_H, IMG_W


class WagonOCRInference:
    def __init__(self, checkpoint_path: str, device: str = "auto",
                 confidence_threshold: float = 0.85):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = confidence_threshold

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt.get("config", {})

        self.model = WagonNumberOCR(
            num_digits=cfg.get("num_digits", 7),
            num_classes=cfg.get("num_classes", 10),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self._tf = A.Compose([
            A.Resize(IMG_H, IMG_W),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])

    # ── Public ────────────────────────────────────────────────────────────────

    def predict(self, image) -> str:
        """np.ndarray или PIL.Image → '1616408'"""
        t = self._prep(image)
        with torch.inference_mode():
            logits = self.model(t)
        return "".join(str(d) for d in logits.argmax(-1)[0].tolist())

    def predict_with_confidence(self, image) -> tuple:
        """Возвращает (prediction, avg_conf, is_reliable)"""
        t = self._prep(image)
        with torch.inference_mode():
            logits = self.model(t)
        probs   = torch.softmax(logits, dim=-1)
        avg_conf = probs.max(-1).values.mean().item()
        pred     = "".join(str(d) for d in logits.argmax(-1)[0].tolist())
        return pred, round(avg_conf, 4), avg_conf >= self.threshold

    def predict_batch(self, images: list) -> list:
        tensors = torch.cat([self._prep(img) for img in images], dim=0)
        with torch.inference_mode():
            logits = self.model(tensors)
        return ["".join(str(d) for d in row.tolist()) for row in logits.argmax(-1)]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prep(self, image) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = np.array(image.convert("L"))
        elif isinstance(image, np.ndarray) and image.ndim == 3:
            image = image.mean(axis=2).astype(np.uint8)
        return self._tf(image=image)["image"].unsqueeze(0).to(self.device)
