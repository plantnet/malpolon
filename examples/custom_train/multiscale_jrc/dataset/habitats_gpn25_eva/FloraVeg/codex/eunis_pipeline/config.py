"""Typed representation and YAML loader for a reproducible experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


EunisLevel = Literal["1", "2", "3", "4", "3_4"]
ALL_LEVELS: tuple[EunisLevel, ...] = ("1", "2", "3", "4", "3_4")


@dataclass(frozen=True)
class DataConfig:
    """Location and column conventions for the FloraVeg metadata tables."""
    train_csv: Path
    test_csv: Path
    image_dir: Path
    site_id_column: str = "id_floraveg"
    image_column: str = "filename_photos"
    latitude_column: str = "lat"
    longitude_column: str = "lon"
    use_gps: bool = False
    validation_fraction: float = 0.10
    split_seed: int = 42
    subset_fraction: float = 1.0  # Use a fraction of the dataset for debugging (1.0 = full dataset)


@dataclass(frozen=True)
class FreezeConfig:
    """How much of a backbone to freeze."""
    freeze: bool = False
    granularity: Literal["blocks", "layers"] = "blocks"
    unfreeze_N: int = 0  # N=0: no freezing, N=1: freeze up to last layer, N=2: up to second-last, etc.


@dataclass(frozen=True)
class ModelConfig:
    """Image encoder and optional GPS-fusion settings."""
    name_img: str = "resnet18"
    name_gps: str | None = None
    pretrained: bool = False  # never downloads weights; only locally cached weights may be used
    gps_embedding_dim: int = 32
    fusion: Literal["concat", "mean"] = "concat"
    freeze_img_backbone: FreezeConfig = field(default_factory=FreezeConfig)
    freeze_gps_backbone: FreezeConfig = field(default_factory=FreezeConfig)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings and one output cardinality for every EUNIS level."""
    levels: tuple[EunisLevel, ...] = ("2", "3_4")
    class_counts: dict[str, int] = field(default_factory=lambda: {"1": 9, "2": 35, "3": 209, "4": 11, "3_4": 215})
    level_weights: tuple[float, ...] = (0.5, 0.85)
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_workers: int = 4
    label_smoothing: float = 0.0
    device: str = "auto"

    def __post_init__(self) -> None:
        """Reject inconsistent head/weight declarations before model construction."""
        if not self.levels or any(level not in ALL_LEVELS for level in self.levels):
            raise ValueError(f"levels must be chosen from {ALL_LEVELS}")
        if len(self.levels) != len(self.level_weights):
            raise ValueError("levels and level_weights must have the same length")
        if any(self.class_counts[level] <= 0 for level in self.levels):
            raise ValueError("each selected level needs a positive class count")


@dataclass(frozen=True)
class PipelineConfig:
    """All persistent settings required to train, evaluate, and export a model."""
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path("outputs/eunis_multihead")


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution-mode switches which do not define the model architecture."""
    resume: bool = False
    evaluate_only: bool = False


def _resolve_path(value: str, config_path: Path) -> Path:
    """Resolve YAML-relative paths relative to the configuration file itself."""
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_pipeline_config(config_path: Path) -> tuple[PipelineConfig, RuntimeConfig]:
    """Load and type-check the complete experiment described by ``config.yaml``.

    Relative paths deliberately resolve against the YAML parent, rather than the
    caller's working directory. This permits the same configuration to be used
    from notebooks, schedulers, and direct shell invocations.
    """
    config_path = Path(config_path).resolve()
    try:
        payload: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("config.yaml must contain a top-level mapping")
    required_sections = {"data", "model", "training", "runtime"}
    missing_sections = required_sections - payload.keys()
    if missing_sections:
        raise ValueError(f"config.yaml is missing sections: {sorted(missing_sections)}")
    data, model, training, runtime = (payload[name] for name in ("data", "model", "training", "runtime"))
    if not all(isinstance(section, dict) for section in (data, model, training, runtime)):
        raise ValueError("data, model, training, and runtime must each be mappings")
    try:
        # Construct each dataclass explicitly so required YAML keys fail early
        # with a readable message rather than much later in a training epoch.
        data_config = DataConfig(
            train_csv=_resolve_path(data["train_csv"], config_path),
            test_csv=_resolve_path(data["test_csv"], config_path),
            image_dir=_resolve_path(data["image_dir"], config_path),
            **{key: value for key, value in data.items() if key not in {"train_csv", "test_csv", "image_dir"}},
        )
        training_config = TrainingConfig(
            levels=tuple(training["levels"]),
            class_counts=training["class_counts"],
            level_weights=tuple(training["level_weights"]),
            **{key: value for key, value in training.items() if key not in {"levels", "class_counts", "level_weights"}},
        )
        pipeline = PipelineConfig(
            data=data_config,
            model=ModelConfig(**model),
            training=training_config,
            output_dir=_resolve_path(payload.get("output_dir", "outputs/eunis_multihead"), config_path),
        )
        return pipeline, RuntimeConfig(**runtime)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid configuration in {config_path}: {error}") from error
