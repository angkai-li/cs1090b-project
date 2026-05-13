"""
Loss functions for class-imbalanced classification (FocalLoss).
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary classification (Lin et al. RetinaNet 2017).

        FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    where pt is the predicted probability of the true class.

    Two mechanisms to address class imbalance:
      - alpha_t: per-class weight. alpha_pos applies to the positive (label=1) class,
        (1 - alpha_pos) to the negative class. For rare-positive imbalance (e.g., 12.3%
        boundary rate for monthly boundary detection), alpha_pos > 0.5 up-weights the
        positive class.
      - (1 - pt)^gamma: focal modulator. Down-weights well-classified examples
        (high pt) so the model focuses on hard cases regardless of class.

    Default alpha_pos=0.75, gamma=2.0 follows the RetinaNet paper and is a reasonable
    starting point for moderate-to-severe binary imbalance.
    """

    def __init__(self, alpha_pos=0.75, gamma=2.0):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,) long
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)  # probability of true class
        alpha_t = torch.where(targets == 1, self.alpha_pos, 1.0 - self.alpha_pos).to(ce.dtype)
        focal_weight = (1.0 - pt) ** self.gamma
        return (alpha_t * focal_weight * ce).mean()
