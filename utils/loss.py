"""utils/loss.py"""

import torch
import torch.nn as nn


class WagonOCRLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.05, ordinal_weight: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ordinal_weight = ordinal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """logits: (B,7,10)  targets: (B,7)"""
        ce_loss = sum(self.ce(logits[:, i, :], targets[:, i]) for i in range(7))

        if self.ordinal_weight > 0:
            preds = logits.detach().argmax(dim=-1).float()
            ordinal = (preds - targets.float()).abs().mean()
            total = ce_loss + self.ordinal_weight * ordinal
        else:
            ordinal = torch.tensor(0.0)
            total = ce_loss

        return total, {
            "loss_ce": ce_loss.item(),
            "loss_ordinal": ordinal.item() if isinstance(ordinal, torch.Tensor) else 0.0,
            "loss_total": total.item(),
        }
