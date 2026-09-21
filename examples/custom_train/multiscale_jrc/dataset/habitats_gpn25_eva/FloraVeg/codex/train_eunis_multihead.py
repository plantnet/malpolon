#!/usr/bin/env python3
"""Train and evaluate the offline multi-head EUNIS habitat classifier.

The script intentionally has no command-line options.  It loads the adjacent
``config.yaml`` so that one human-readable file records a fully reproducible
experiment, including paths, model settings, and runtime mode.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from time import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eunis_pipeline.config import load_pipeline_config
from eunis_pipeline.data import (
    EunisMultiLabelDataset,
    observed_class_counts,
    read_metadata,
    split_train_validation,
)
from eunis_pipeline.engine import run_epoch
from eunis_pipeline.models import MultiHeadEunisClassifier, build_image_encoder, build_gps_encoder
from eunis_pipeline.reporting import save_epoch_metrics, save_test_artifacts
from eunis_pipeline.transforms import make_transforms


def export_predictions(
    model, loader, dataset, device, levels, output_path: Path
) -> None:
    """Write one decoded prediction record per test site.

    ``valid_prediction`` is true when the top-1 class is one of the possible
    reference labels for that site. This is the same soft multi-label convention
    used by the evaluation metric.
    """
    model.eval()
    records: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["image"].to(device),
                batch.get("gps").to(device) if batch.get("gps") is not None else None,
            )
            for index, site_id in enumerate(batch["site_id"]):
                row = {"id_floraveg": int(site_id)}
                for level in levels:
                    probabilities = outputs[level][index].softmax(0)
                    prediction = int(probabilities.argmax())
                    row[f"pred_label_encoded_lvl{level}"] = prediction
                    row[f"pred_label_lvl{level}"] = str(dataset.label_to_code[level].get(prediction, "<unknown>"))
                    row[f"pred_confidence_lvl{level}"] = float(
                        probabilities[prediction]
                    )
                    row[f"valid_prediction_lvl{level}"] = bool(
                        batch["targets"][level][index, prediction]
                    )
                records.append(row)
    predictions = pd.DataFrame(records)
    metadata = dataset.frame.drop_duplicates(
        subset=[dataset.data_config.site_id_column]
    )
    predictions = metadata.merge(
        predictions,
        how="left",
        left_on=dataset.data_config.site_id_column,
        right_on="id_floraveg",
    )
    predictions.to_csv(output_path, index=False)


def info(message: str) -> None:
    """Print a consistently prefixed, immediately visible pipeline status message."""
    print(f"[INFO] {message}", flush=True)


def main() -> None:
    """Build the configured experiment, optionally train it, then test/export it."""
    # Keeping the YAML next to the entry point makes the invocation independent
    # of the current working directory.
    config_path = Path(__file__).with_name("config.yaml")
    config, runtime = load_pipeline_config(config_path)
    levels = config.training.levels
    info(f"Loaded configuration: {config_path}")
    info(f"Train CSV: {config.data.train_csv}")
    info(f"Test CSV: {config.data.test_csv}")
    modalities = "image + GPS coordinates" if config.data.use_gps else "image"
    info(f"Modalities: {modalities}")
    info(f"Model: img_backbone={config.model.name_img}; gps_backbone={config.model.name_gps}; EUNIS levels: {', '.join(levels)}")
    if config.data.use_gps:
        info(f"Fusion strategy: {config.model.fusion}")
    info(f"Requested EUNIS levels: {', '.join(levels)}")
    info(f"Configured class counts: {config.training.class_counts}")
    info(f"Batch size: {config.training.batch_size}")
    info(f"Epochs: {config.training.epochs}")
    info(f"Runtime mode: resume={runtime.resume}, evaluate_only={runtime.evaluate_only}")

    info("Creating output directory and saving the resolved configuration")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the resolved configuration alongside artifacts for provenance.
    (config.output_dir / "config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2)
    )
    info("Selecting compute device")
    device = torch.device(
        "cuda"
        if config.training.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.training.device == "auto"
        else config.training.device
    )
    info(f"Device: {device}")
    
    # ----------------------------------------------------------------------
    # Loading data
    # ----------------------------------------------------------------------

    info("Reading and validating train/test metadata")
    train_frame = read_metadata(config.data.train_csv, levels, config.data)
    test_frame = read_metadata(config.data.test_csv, levels, config.data)
    train_class_counts = observed_class_counts(train_frame, levels)
    test_class_counts = observed_class_counts(test_frame, levels)
    info(
        "Train metadata: "
        f"rows={len(train_frame)}, observed unique classes per EUNIS level={train_class_counts}"
    )
    info(
        "Test metadata: "
        f"rows={len(test_frame)}, observed unique classes per EUNIS level={test_class_counts}"
    )
    # Validation is derived only from the declared training CSV; the test CSV is
    # never used to choose checkpoints or tune the model.
    info("Creating deterministic train/validation split")
    train_frame, validation_frame = split_train_validation(
        train_frame, levels[-1], config.data.validation_fraction, config.data.split_seed
    )
    info(
        f"Split sizes: train={len(train_frame)}, validation={len(validation_frame)}, test={len(test_frame)}"
    )
    info("Building model-specific training and evaluation transforms")
    train_tf, evaluation_tf = make_transforms(config.model.name_img)
    
    # ----------------------------------------------------------------------
    # Datasets & Datalaoders
    # ----------------------------------------------------------------------

    info("Creating train, validation, and test datasets")
    # Instantiation of train, val & test datasets with transforms applied
    datasets = [
        EunisMultiLabelDataset(
            frame,
            config.data.image_dir,
            levels,
            config.training.class_counts,
            transform,
            config.data,
        )
        for frame, transform in (
            (train_frame, train_tf),
            (validation_frame, evaluation_tf),
            (test_frame, evaluation_tf),
        )
    ]
    # Pinning host memory is useful only when CUDA asynchronously copies batches.
    info("Creating DataLoaders")
    train_loader, validation_loader, test_loader = [
        DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            num_workers=config.training.num_workers,
            pin_memory=device.type == "cuda",
        )
        for dataset, shuffle in zip(datasets, (True, False, False))
    ]
    info(
        "Dataset sizes after one-record-per-site deduplication: "
        f"train={len(datasets[0])}, validation={len(datasets[1])}, test={len(datasets[2])}"
    )

    # ----------------------------------------------------------------------
    # Models, optimizers
    # ----------------------------------------------------------------------

    info("Building shared image encoder and EUNIS classification heads")
    img_encoder = build_image_encoder(config.model.name_img, config.model.pretrained)
    if config.data.use_gps:
        gps_encoder = build_gps_encoder(config.model.name_gps, config.model.gps_embedding_dim)
    else:
        gps_encoder = None
    model = MultiHeadEunisClassifier(img_encoder, gps_encoder, config.model.name_img, config.model.name_gps,
                                     config.training.class_counts, levels,
                                     config.data.use_gps, config.model.fusion,
                                     config.model.freeze_img_backbone, config.model.freeze_gps_backbone).to(device)
    info("Creating Adam optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    best_path, last_path = (
        config.output_dir / "best_model.pt",
        config.output_dir / "last_model.pt",
    )
    start_epoch, best_loss = 0, float("inf")
    if runtime.resume:
        # Resume optimizer state as well as parameters so Adam moments are kept.
        info(f"Resuming checkpoint: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_loss = checkpoint["epoch"] + 1, checkpoint["best_loss"]

    # ----------------------------------------------------------------------
    # Training & validation loop
    # ----------------------------------------------------------------------

    if not runtime.evaluate_only:
        info("Starting training and validation")
        epoch_progress = tqdm(
            range(start_epoch, config.training.epochs),
            desc="Training epochs",
            unit="epoch",
        )
        for epoch in epoch_progress:
            info(f"Starting epoch {epoch}")
            info("  > Training")
            train_result = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                levels,
                config.training.class_counts,
                config.training.level_weights,
                config.training.label_smoothing,
                split_name="train",
            )
            save_epoch_metrics(
                config.output_dir / "train_metrics.csv", epoch, "train", train_result
            )

            info("  > Validation")
            validation_result = run_epoch(
                model,
                validation_loader,
                None,
                device,
                levels,
                config.training.class_counts,
                config.training.level_weights,
                config.training.label_smoothing,
                split_name="validation",
            )
            save_epoch_metrics(
                config.output_dir / "validation_metrics.csv",
                epoch,
                "validation",
                validation_result,
            )

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, last_path)
            if validation_result.loss < best_loss:
                best_loss = validation_result.loss
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, best_path)
            info(
                f"Epoch {epoch} complete: train_loss={train_result.loss:.4f}, "
                f"validation_loss={validation_result.loss:.4f}, "
                f"validation_accuracy={validation_result.accuracy}"
            )
            epoch_progress.set_postfix(
                train_loss=f"{train_result.loss:.4f}",
                validation_loss=f"{validation_result.loss:.4f}",
            )
    if not best_path.exists():
        raise FileNotFoundError(
            f"No best checkpoint at {best_path}; train first or use --resume"
        )

    # ----------------------------------------------------------------------
    # Test evaluation and export predictions
    # ----------------------------------------------------------------------

    # Evaluation always uses the validation-selected checkpoint, not last epoch.
    info(f"Loading best validation checkpoint: {best_path}")
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["model"]
    )
    info("Running final test evaluation")
    test_result = run_epoch(
        model,
        test_loader,
        None,
        device,
        levels,
        config.training.class_counts,
        config.training.level_weights,
        config.training.label_smoothing,
        split_name="test",
    )
    info("Saving test metrics and confusion matrices")
    save_test_artifacts(config.output_dir, test_result)
    info("Exporting decoded test predictions")
    export_predictions(
        model,
        test_loader,
        datasets[2],
        device,
        levels,
        config.output_dir / "test_predictions.csv",
    )
    info(
        f"Test complete: loss={test_result.loss:.4f}; "
        f"accuracy={test_result.accuracy}; outputs={config.output_dir}"
    )
    print(f'End time: {t1-time()}s')


if __name__ == "__main__":
    t1 = time()
    print(f'Start time: {t1}')
    main()
