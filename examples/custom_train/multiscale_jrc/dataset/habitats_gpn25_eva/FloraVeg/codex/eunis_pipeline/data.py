"""Metadata schema validation and PyTorch dataset implementation."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import ALL_LEVELS, DataConfig


def level_suffix(level: str) -> str:
    """Return the FloraVeg column suffix for an EUNIS prediction level.

    The merged level-3/4 label is the historical unsuffixed representation.
    """
    return "" if level == "3_4" else f"_lvl{level}"


def required_columns(levels: tuple[str, ...], config: DataConfig) -> set[str]:
    """List the CSV columns needed for the selected heads and modalities."""
    columns = {config.site_id_column, config.image_column}
    for level in levels:
        suffix = level_suffix(level)
        columns.update({f"habitats_code{suffix}", f"habitats_code_ID{suffix}", f"habitats_code_ID_oh{suffix}"})
    if config.use_gps:
        columns.update({config.latitude_column, config.longitude_column})
    return columns


def read_metadata(path: Path, levels: tuple[str, ...], config: DataConfig) -> pd.DataFrame:
    """Read a metadata CSV and fail before training if its schema is incomplete."""
    frame = pd.read_csv(path)
    frame = frame.sample(frac=config.subset_fraction, random_state=config.split_seed).reset_index(drop=True)
    missing = required_columns(levels, config) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def split_train_validation(frame: pd.DataFrame, level: str, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic stratified split while retaining singletons in train.

    Multi-label combinations are stratified as their semicolon-separated encoded
    string. Every non-singleton group supplies at least one validation sample and
    retains at least one training sample.
    """
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be strictly between zero and one")
    label_column = f"habitats_code_ID{level_suffix(level)}"
    generator = torch.Generator().manual_seed(seed)
    validation_indices: list[int] = []
    # Group-wise sampling mirrors a stratified split while allowing all singleton
    # habitats to remain available to the training objective.
    for _, group in frame.groupby(label_column, sort=False):
        if len(group) < 2:
            continue
        count = min(len(group) - 1, max(1, round(len(group) * fraction)))
        choices = torch.randperm(len(group), generator=generator)[:count].tolist()
        validation_indices.extend(group.index[position] for position in choices)
    validation = frame.loc[validation_indices]
    training = frame.drop(index=validation_indices)
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def parse_one_hot(value: object, class_count: int) -> torch.Tensor:
    """Parse the legacy ``[0 1 ...]`` CSV encoding into a float target vector."""
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text.replace(" ", ",")) if "," not in text else ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text.strip("[]").split()
    else:
        parsed = value
    tensor = torch.as_tensor(parsed, dtype=torch.float32).flatten()
    if tensor.numel() != class_count:
        raise ValueError(f"one-hot label has {tensor.numel()} entries; expected {class_count}")
    return tensor


def parse_label_ids(value: object) -> list[int]:
    """Split a semicolon-separated encoded label cell into integer class IDs."""
    return [int(part) for part in str(value).split(";")]


def observed_class_counts(frame: pd.DataFrame, levels: tuple[str, ...]) -> dict[str, int]:
    """Count distinct encoded classes observed for every requested EUNIS level.

    A metadata cell may contain several valid labels, such as ``"26;58"``.
    Every semicolon-separated encoded ID is counted independently, rather than
    treating the entire string as a single composite class.
    """
    counts: dict[str, int] = {}
    for level in levels:
        column = f"habitats_code_ID{level_suffix(level)}"
        labels = {
            label_id
            for value in frame[column]
            for label_id in parse_label_ids(value)
        }
        counts[level] = len(labels)
    return counts


def _coordinate(value: object) -> float:
    """Accept numeric cells and legacy semicolon-separated repeated metadata."""
    try:
        return float(str(value).split(";")[0])
    except (ValueError, TypeError):
        return float("nan")


class EunisMultiLabelDataset(Dataset):
    """Serve one RGB image/site and a multi-hot target per requested EUNIS level.

    Repeated image rows for a single ``id_floraveg`` are collapsed because the
    supplied multi-label CSVs already associate all valid habitat codes with the
    site. When enabled, the returned ``gps`` tensor is ordered ``[latitude,
    longitude]``.
    """

    def __init__(self, frame: pd.DataFrame, image_dir: Path, levels: tuple[str, ...], class_counts: Mapping[str, int],
                 transform=None, data_config: DataConfig | None = None):
        self.frame = frame.drop_duplicates(subset=[data_config.site_id_column]).reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.levels = levels
        self.class_counts = class_counts
        self.transform = transform
        self.data_config = data_config
        self.label_to_code = self._build_label_table()

    def _build_label_table(self) -> dict[str, dict[int, str]]:
        """Map encoded target indices back to EUNIS codes for prediction exports."""
        tables = {level: {} for level in self.levels}
        for level in self.levels:
            suffix = level_suffix(level)
            for ids, codes in zip(self.frame[f"habitats_code_ID{suffix}"], self.frame[f"habitats_code{suffix}"]):
                tables[level].update(zip(parse_label_ids(ids), str(codes).split(";")))
        return tables

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Load one image and build the dictionary collated by ``DataLoader``."""
        row = self.frame.iloc[index]
        image_path = self.image_dir / str(row[self.data_config.image_column]).strip()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        targets = {
            level: parse_one_hot(
                row[f"habitats_code_ID_oh{level_suffix(level)}"], self.class_counts[level]
            )
            for level in self.levels
        }
        item: dict[str, object] = {"image": image, "targets": targets, "site_id": row[self.data_config.site_id_column]}
        if self.data_config.use_gps:
            item["gps"] = torch.tensor(
                [_coordinate(row[self.data_config.latitude_column]), _coordinate(row[self.data_config.longitude_column])],
                dtype=torch.float32,
            )
        return item
