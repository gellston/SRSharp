import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: sqrt((x-y)^2 + eps^2)
    - L1보다 매끈하고(outlier에 강함) SR에서 자주 씀
    """
    def __init__(self, eps=1e-3, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + (self.eps * self.eps))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss