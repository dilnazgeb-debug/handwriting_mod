"""
models/model.py
───────────────
Лёгкая модель для маленького датасета (< 200 примеров).

Стратегия:
- MobileNetV3-Small backbone (меньше параметров чем EfficientNet)
- Backbone заморожен первые N эпох → учим только головы
- Никакого LSTM (мало данных — LSTM переобучается)
- 7 простых FC-голов напрямую из CNN features
"""

import torch
import torch.nn as nn
from torchvision import models


class WagonNumberOCR(nn.Module):
    def __init__(self, num_digits: int = 7, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        self.num_digits = num_digits

        # MobileNetV3-Small: быстрый, лёгкий, хорошо работает на малых датасетах
        backbone = models.mobilenet_v3_small(weights="DEFAULT")
        # Убираем classifier, оставляем features + avgpool
        self.features = backbone.features   # → (B, 576, H', W')
        self.pool     = nn.AdaptiveAvgPool2d(1)  # → (B, 576, 1, 1)

        feat_dim = 576

        # Shared neck: общий для всех голов
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 7 независимых голов
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, num_classes),
            )
            for _ in range(num_digits)
        ])

        # Инициализация голов
        for head in self.heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Заморозить backbone — учим только головы."""
        for p in self.features.parameters():
            p.requires_grad = False
        print("  [backbone frozen]")

    def unfreeze_backbone(self):
        """Разморозить backbone для fine-tuning."""
        for p in self.features.parameters():
            p.requires_grad = True
        print("  [backbone unfrozen — fine-tuning]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → logits: (B, 7, 10)"""
        x = x.repeat(1, 3, 1, 1)                    # grayscale → 3ch
        x = self.features(x)                          # (B, 576, H', W')
        x = self.pool(x).flatten(1)                   # (B, 576)
        x = self.neck(x)                              # (B, 256)

        return torch.stack(
            [self.heads[i](x) for i in range(self.num_digits)], dim=1
        )                                             # (B, 7, 10)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> list:
        preds = self.forward(x).argmax(dim=-1)
        return ["".join(str(d.item()) for d in row) for row in preds]
