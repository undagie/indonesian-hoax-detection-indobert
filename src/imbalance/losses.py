"""Imbalance losses: Focal, Class-Balanced, LDAM, and class-weighted focal (combined)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with optional per-class alpha vector."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Sequence[float]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - target_probs).pow(self.gamma)
        loss = -focal * target_log_probs
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)[targets]
            loss = alpha * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """Effective-number reweighting (Cui et al., 2019).

    Wraps any base loss ("ce" or "focal"). Weights are computed from the
    per-class counts of the *training set*.
    """

    def __init__(
        self,
        samples_per_class: Sequence[int],
        beta: float = 0.999,
        base_loss: str = "focal",
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        eff_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.base_loss = base_loss
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(logits.device)
        if self.base_loss == "ce":
            return F.cross_entropy(logits, targets, weight=w)
        if self.base_loss == "focal":
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_probs = target_log_probs.exp()
            focal = (1.0 - target_probs).pow(self.gamma)
            alpha = w[targets]
            return -(alpha * focal * target_log_probs).mean()
        raise ValueError(f"unknown base_loss {self.base_loss!r}")


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss (Cao et al., 2019)."""

    def __init__(
        self,
        samples_per_class: Sequence[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.maximum(samples_per_class, 1)))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = float(s)
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        batch_m = self.m_list.to(logits.device)[targets].view(-1, 1)
        logits_m = logits - index.float() * batch_m
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(self.s * logits_m, targets, weight=w)


def build_loss(
    name: str,
    *,
    num_classes: int,
    samples_per_class: Optional[Sequence[int]] = None,
    class_weights: Optional[Sequence[float]] = None,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
    cb_beta: float = 0.999,
    ldam_max_m: float = 0.5,
    ldam_s: float = 30.0,
) -> nn.Module:
    """Factory matching ``IMBALANCE_STRATEGIES`` in config.py."""
    name = name.lower()
    if name in {"none", "class_weight", "weighted_sampler", "random_oversample",
                "random_undersample", "smote_embed"}:

        weight = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None and name == "class_weight"
            else None
        )
        return nn.CrossEntropyLoss(weight=weight)
    if name == "focal_loss":
        alpha = [focal_alpha] * num_classes if focal_alpha is not None else None
        return FocalLoss(gamma=focal_gamma, alpha=alpha)
    if name == "cb_loss":
        assert samples_per_class is not None, "CB loss needs samples_per_class"
        return ClassBalancedLoss(samples_per_class, beta=cb_beta, base_loss="focal",
                                 gamma=focal_gamma)
    if name == "ldam_loss":
        assert samples_per_class is not None, "LDAM loss needs samples_per_class"
        return LDAMLoss(samples_per_class, max_m=ldam_max_m, s=ldam_s,
                        weight=class_weights)
    if name == "combined":


        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    raise ValueError(f"unknown imbalance strategy {name!r}")
