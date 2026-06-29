"""utils/metrics.py"""

import torch


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    preds = logits.argmax(dim=-1)
    return {
        "digit_accuracy":    (preds == targets).float().mean().item(),
        "sequence_accuracy": (preds == targets).all(dim=1).float().mean().item(),
    }
