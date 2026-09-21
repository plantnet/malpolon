#!/usr/bin/env python3
"""Same-node multi-GPU EUNIS training with :class:`torch.nn.DataParallel`.

Run this file directly after setting ``CUDA_VISIBLE_DEVICES`` to the GPUs that
may be used. It reads ``config_dpl.yaml`` exactly like ``train_eunis_multihead.py``
but writes artifacts to ``<output_dir>/dataparallel`` to avoid overwriting a
single-GPU experiment.

``DataParallel`` is intentionally a simple compatibility option. For sustained
multi-GPU training, ``train_eunis_multihead_ddp.py`` is generally faster and
uses one process per GPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eunis_pipeline.config import load_pipeline_config
from eunis_pipeline.data import EunisMultiLabelDataset, observed_class_counts, read_metadata, split_train_validation
from eunis_pipeline.engine import run_epoch
from eunis_pipeline.models import MultiHeadEunisClassifier, build_image_encoder, build_gps_encoder
from eunis_pipeline.reporting import save_epoch_metrics, save_test_artifacts
from eunis_pipeline.transforms import make_transforms
from train_eunis_multihead import export_predictions, info

### DEBUG
# For some unknown reason, the loss epxlodes to infinity where using DP with multiple GPUs.
# Forcing device_ids=[0] seems to fix the problem, but this is not a real solution.
# Debugging session with chatGPT: https://chatgpt.com/share/6aa90dad-7ef8-83eb-85cf-53ecfe85d93c
# Data is not corrupted, no AMP, no BatchNorm, no in-place operation, no wrong assignment of device
import torch
def check_params(model):
  bad_params, bad_buffers = [], []
  for name, param in model.named_parameters():
      if not torch.isfinite(param).all():
          bad_params.append(param)
          print("param bad:", name, torch.isnan(param).any(), torch.isinf(param).any())
  for name, buf in model.named_buffers():
      if not torch.isfinite(buf).all():
          bad_buffers.append(buf)
          print("buffer bad:", name, torch.isnan(buf).any(), torch.isinf(buf).any())
  return len(bad_params), len(bad_buffers)


# To put in engine.py:
# ### debug
# check_model(model, "before forward")
# ### debug
# outputs = model(images, gps)
# ### debug
# check_model(model, "after forward")
# ### debug
# loss = sum(
#     weight * soft_multilabel_cross_entropy(outputs[level], batch_targets[level], label_smoothing)
#     for level, weight in zip(levels, level_weights)
# )
# ### debug
# print("loss finite:", torch.isfinite(loss).all().item())
# ### debug
# if training:
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     ### debug
#     check_model(model, "after backward")
#     for name, p in model.named_parameters():
#         if p.grad is not None and not torch.isfinite(p.grad).all():
#             print("BAD GRAD:", name)
#     ### debug
#     optimizer.step()
#     ### debug
#     check_model(model, "after optimizer")
#     ### debug

# To put in engine.py:
# import torch
# def check_params(model):
#   bad_params, bad_buffers = [], []
#   for name, param in model.named_parameters():
#       if not torch.isfinite(param).all():
#           bad_params.append(param)
#           print("param bad:", name, torch.isnan(param).any(), torch.isinf(param).any())
#   for name, buf in model.named_buffers():
#       if not torch.isfinite(buf).all():
#           bad_buffers.append(buf)
#           print("buffer bad:", name, torch.isnan(buf).any(), torch.isinf(buf).any())
#   return len(bad_params), len(bad_buffers)

# def check_model(model, tag):
#     for name, p in model.named_parameters():
#         if not torch.isfinite(p).all():
#             print(f"[{tag}] NONFINITE PARAM: {name}")
#             print("min:", p.nan_to_num().min().item())
#             print("max:", p.nan_to_num().max().item())
#             return False
#     return True
### debug

def checkpoint_model(model: nn.Module) -> nn.Module:
    """Return the unwrapped model so checkpoints work with non-parallel loading."""
    return model.module if isinstance(model, nn.DataParallel) else model


def main() -> None:
    """Train one shared model replicated across all visible CUDA devices."""
    config_path = Path(__file__).with_name("config_dp.yaml")
    config, runtime = load_pipeline_config(config_path)
    levels = config.training.levels
    if not torch.cuda.is_available():
        raise RuntimeError("DataParallel requires CUDA and at least two visible GPUs.")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(f"DataParallel requires at least two visible GPUs; found {gpu_count}.")
    device = torch.device("cuda:0")
    output_dir = config.output_dir  #  / "dataparallel"

    info(f"[DataParallel] visible GPUs: {gpu_count}; primary device: {device}")
    info(f"[DataParallel] output directory: {output_dir}")
    info(f"Train CSV: {config.data.train_csv}")
    info(f"Test CSV: {config.data.test_csv}")
    info(f"Model: img_backbone={config.model.name_img}; gps_backbone={config.model.name_gps}; EUNIS levels: {', '.join(levels)}")
    info(f"Batch size (global): {config.training.batch_size}; epochs: {config.training.epochs}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), default=str, indent=2))

    info("Reading metadata and counting semicolon-separated encoded labels")
    train_frame = read_metadata(config.data.train_csv, levels, config.data)
    test_frame = read_metadata(config.data.test_csv, levels, config.data)
    info(f"Observed train classes: {observed_class_counts(train_frame, levels)}")
    info(f"Observed test classes: {observed_class_counts(test_frame, levels)}")
    train_frame, validation_frame = split_train_validation(train_frame, levels[-1], config.data.validation_fraction, config.data.split_seed)

    info("Building transforms, datasets, and DataLoaders")
    train_tf, evaluation_tf = make_transforms(config.model.name_img)
    datasets = [
        EunisMultiLabelDataset(frame, config.data.image_dir, levels, config.training.class_counts, transform, config.data)
        for frame, transform in ((train_frame, train_tf), (validation_frame, evaluation_tf), (test_frame, evaluation_tf))
    ]
    train_loader, validation_loader, test_loader = [
        DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=config.training.num_workers, pin_memory=True)
        for dataset, shuffle in zip(datasets, (True, False, False))
    ]

    info("Constructing and replicating the model with torch.nn.DataParallel")
    img_encoder = build_image_encoder(config.model.name_img, config.model.pretrained)
    gps_encoder = build_gps_encoder(config.model.name_gps, config.model.gps_embedding_dim)
    base_model = MultiHeadEunisClassifier(img_encoder, gps_encoder, config.training.class_counts, levels,
                                          config.data.use_gps, config.model.fusion,
                                          config.model.freeze_img_backbone, config.model.freeze_gps_backbone).to(device)
    model = nn.DataParallel(base_model, device_ids=[1], output_device=0)# list(range(gpu_count)), output_device=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    best_path, last_path = output_dir / "best_model.pt", output_dir / "last_model.pt"
    start_epoch, best_loss = 0, float("inf")
    if runtime.resume:
        info(f"Resuming checkpoint: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        checkpoint_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_loss = checkpoint["epoch"] + 1, checkpoint["best_loss"]

    print("DataParallel devices:", model.device_ids)
    print("Output device:", model.output_device)
    print("Model device:", next(model.parameters()).device)

    if not runtime.evaluate_only:
        for epoch in tqdm(range(start_epoch, config.training.epochs), desc="DataParallel epochs", unit="epoch"):
            info(f"Starting epoch {epoch}")
            train_result = run_epoch(model, train_loader, optimizer, device, levels, config.training.class_counts, config.training.level_weights, config.training.label_smoothing, split_name="train")
            validation_result = run_epoch(model, validation_loader, None, device, levels, config.training.class_counts, config.training.level_weights, config.training.label_smoothing, split_name="validation")
            save_epoch_metrics(output_dir / "train_metrics.csv", epoch, "train", train_result)
            save_epoch_metrics(output_dir / "validation_metrics.csv", epoch, "validation", validation_result)
            checkpoint = {"epoch": epoch, "model": checkpoint_model(model).state_dict(), "optimizer": optimizer.state_dict(), "best_loss": best_loss}
            torch.save(checkpoint, last_path)
            if validation_result.loss < best_loss:
                best_loss = validation_result.loss
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, best_path)
            info(f"Epoch {epoch} complete: train_loss={train_result.loss:.4f}, validation_loss={validation_result.loss:.4f}")

    if not best_path.exists():
        raise FileNotFoundError(f"No best checkpoint at {best_path}; train first or enable resume.")
    info(f"Loading best checkpoint and evaluating: {best_path}")
    checkpoint_model(model).load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    test_result = run_epoch(model, test_loader, None, device, levels, config.training.class_counts, config.training.level_weights, config.training.label_smoothing, split_name="test")
    save_test_artifacts(output_dir, test_result)
    export_predictions(model, test_loader, datasets[2], device, levels, output_dir / "test_predictions.csv")
    info(f"DataParallel test complete: loss={test_result.loss:.4f}, accuracy={test_result.accuracy}")


if __name__ == "__main__":
    main()
