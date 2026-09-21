"""One-epoch optimisation and evaluation routines shared by all splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import soft_confusion_matrix, soft_multilabel_cross_entropy, top1_soft_multilabel_accuracy


@dataclass
class EpochResult:
    """Aggregate loss, accuracy, and confusion matrices produced for one split."""
    loss: float
    accuracy: dict[str, float]
    confusion_matrices: dict[str, tuple[np.ndarray, np.ndarray]]


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device,
              levels: tuple[str, ...], class_counts: dict[str, int], level_weights: tuple[float, ...], label_smoothing: float,
              split_name: str = "evaluation") -> EpochResult:
    """Run one complete train or evaluation pass over ``loader``.

    Passing an optimizer enables gradients and parameter updates; passing
    ``None`` performs deterministic evaluation under disabled gradients. Loss is
    a weighted sum of the selected EUNIS-head losses. ``split_name`` labels the
    nested batch progress bar.
    """
    training = optimizer is not None
    model.train(training)
    # Keep full-epoch outputs because the soft-label metric is dataset-level.
    targets, probabilities, predictions = ({level: [] for level in levels} for _ in range(3))
    total_loss = 0.0
    batch_progress = tqdm(
        loader,
        desc=f"{split_name} batches",
        unit="batch",
        leave=False,
    )
    for batch_index, batch in enumerate(batch_progress, start=1):
        images = batch["image"].to(device)
        gps = batch.get("gps")
        gps = gps.to(device) if gps is not None else None
        batch_targets = {level: batch["targets"][level].to(device) for level in levels}
        with torch.set_grad_enabled(training):
            outputs = model(images, gps)
            loss = sum(
                weight * soft_multilabel_cross_entropy(outputs[level], batch_targets[level], label_smoothing)
                for level, weight in zip(levels, level_weights)
            )
            print(f"Loss: {loss.item()}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        batch_progress.set_postfix(loss=f"{total_loss / batch_index:.4f}")
        for level in levels:
            probs = outputs[level].softmax(dim=1).detach().cpu().numpy()
            probabilities[level].append(probs)
            predictions[level].append(probs.argmax(axis=1))
            targets[level].append(batch_targets[level].cpu().numpy())
    if not len(loader):
        raise ValueError("DataLoader is empty")
    matrices, accuracy = {}, {}
    for level in levels:
        target = np.concatenate(targets[level])
        probability = np.concatenate(probabilities[level])
        prediction = np.concatenate(predictions[level])
        accuracy[level] = top1_soft_multilabel_accuracy(target, probability)
        matrices[level] = soft_confusion_matrix(target, prediction, class_counts[level])
    return EpochResult(total_loss / len(loader), accuracy, matrices)
