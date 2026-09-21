"""CSV, NumPy, and figure artifacts emitted after training and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import EpochResult


def save_epoch_metrics(path: Path, epoch: int, split: str, result: EpochResult) -> None:
    """Append per-head loss/accuracy rows for one train or validation epoch."""
    rows = [{"epoch": epoch, "split": split, "loss": result.loss, "eunis_level": level, "top1_soft_multilabel_accuracy": accuracy}
            for level, accuracy in result.accuracy.items()]
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


def save_test_artifacts(output_dir: Path, result: EpochResult) -> None:
    """Write final test metrics plus raw/normalised confusion-matrix artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"loss": result.loss, "eunis_level": level, "top1_soft_multilabel_accuracy": accuracy}
            for level, accuracy in result.accuracy.items()]
    pd.DataFrame(rows).to_csv(output_dir / "test_metrics.csv", index=False)
    for level, (matrix, normalized) in result.confusion_matrices.items():
        # Arrays preserve exact values; PNGs offer a quick visual diagnostic.
        np.save(output_dir / f"confusion_matrix_eunis_{level}.npy", matrix)
        np.save(output_dir / f"confusion_matrix_normalized_eunis_{level}.npy", normalized)
        figure, axes = plt.subplots(1, 2, figsize=(16, 8))
        for axis, values, title in zip(axes, (matrix, normalized), ("Confusion matrix", "Row-normalized confusion matrix")):
            image = axis.imshow(values)
            axis.set(title=f"{title}: EUNIS level {level}", xlabel="Predicted class", ylabel="True class")
            figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_matrix_eunis_{level}.png", dpi=150)
        plt.close(figure)
