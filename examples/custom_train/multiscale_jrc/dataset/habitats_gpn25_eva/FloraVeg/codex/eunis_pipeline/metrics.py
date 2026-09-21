"""Metrics appropriate when a site may have multiple valid habitat labels."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def soft_multilabel_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross entropy against a uniform distribution over each sample's valid labels.

    A target such as ``[0, 1, 0, 1]`` becomes ``[0, .5, 0, .5]``. Consequently,
    the model is not penalised for assigning probability to either accepted code.
    """
    target_distribution = targets / targets.sum(dim=1, keepdim=True).clamp_min(1)
    if label_smoothing:
        target_distribution = target_distribution * (1 - label_smoothing) + label_smoothing / logits.size(1)
    return -(target_distribution * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def top1_soft_multilabel_accuracy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the fraction of sites whose top class is one of their valid labels."""
    predictions = probabilities.argmax(axis=1)
    return float(targets[np.arange(len(targets)), predictions].mean())


def soft_confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Build raw and row-normalised confusion matrices for multi-hot targets.

    Each valid reference class contributes one unit to the selected prediction
    column. Empty classes produce NaN normalised rows, making absence explicit.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for target, prediction in zip(targets, predictions):
        matrix[:, prediction] += target
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix, normalized
