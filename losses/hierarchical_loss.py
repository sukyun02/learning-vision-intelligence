"""
Loss Functions
- Standard CrossEntropy
- CutMix-aware CrossEntropy
- Super-Class Hierarchical Loss  (λ=0.8 as per proposal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.cifar100 import FINE_TO_COARSE


# ---------------------------------------------------------------------------
# Super-class projection layer (no learnable params)
# ---------------------------------------------------------------------------

def build_fine_to_coarse_matrix(device='cpu') -> torch.Tensor:
    """
    Returns a (20, 100) binary matrix M where M[c, f] = 1
    if fine class f belongs to coarse class c.
    Used to aggregate fine logits → coarse logits.
    """
    M = torch.zeros(20, 100)
    for fine, coarse in enumerate(FINE_TO_COARSE):
        M[coarse, fine] = 1.0
    return M.to(device)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class HierarchicalLoss(nn.Module):
    """
    L_total = (1 - λ) * CE_fine + λ * CE_coarse
    λ = 0.8 → coarse loss weighted heavily as in the proposal.

    For CutMix batches supply labels_a, labels_b, lam.
    """

    def __init__(self, lam_coarse: float = 0.8, label_smoothing: float = 0.0):
        super().__init__()
        self.lam_coarse      = lam_coarse
        self.label_smoothing = label_smoothing
        self._M              = None   # lazy init on first forward

    def _get_M(self, device):
        if self._M is None or self._M.device != device:
            self._M = build_fine_to_coarse_matrix(device)
        return self._M

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets,
                               label_smoothing=self.label_smoothing)

    def _coarse_logits(self, fine_logits: torch.Tensor) -> torch.Tensor:
        M = self._get_M(fine_logits.device)          # (20, 100)
        # sum fine log-probs over members of each super-class
        log_p = F.log_softmax(fine_logits, dim=-1)   # (B, 100)
        coarse_log_p = torch.mm(log_p, M.T)          # (B, 20)  — log-sum approximation
        # Alternatively: average logit per super-class
        # coarse_log_p = torch.mm(fine_logits, M.T)
        return coarse_log_p

    def forward(self, fine_logits: torch.Tensor,
                labels_a: torch.Tensor,
                labels_b: torch.Tensor = None,
                lam: torch.Tensor = None) -> torch.Tensor:

        device = fine_logits.device
        M = self._get_M(device)
        mapping = torch.tensor(FINE_TO_COARSE, device=device)

        coarse_logits = self._coarse_logits(fine_logits)
        coarse_a = mapping[labels_a]

        is_cutmix = (labels_b is not None) and (lam is not None) and (lam.item() < 1.0)

        if is_cutmix:
            coarse_b = mapping[labels_b]
            # fine loss
            loss_fine = lam * self._ce(fine_logits, labels_a) + \
                        (1 - lam) * self._ce(fine_logits, labels_b)
            # coarse loss
            loss_coarse = lam * F.nll_loss(coarse_logits, coarse_a) + \
                          (1 - lam) * F.nll_loss(coarse_logits, coarse_b)
        else:
            loss_fine   = self._ce(fine_logits, labels_a)
            loss_coarse = F.nll_loss(coarse_logits, coarse_a)

        return (1 - self.lam_coarse) * loss_fine + self.lam_coarse * loss_coarse
