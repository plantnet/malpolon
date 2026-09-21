#!/usr/bin/env python3
"""Same-node multi-GPU EUNIS training with ``DistributedDataParallel``, launched
*without* ``torchrun``.

This is a self-launching variant of ``codex/train_eunis_multihead_ddp.py``.
The original script relies on ``torchrun`` to fork one OS process per GPU and
export ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` for it to read. Here, ``main()``
instead detects the number of visible CUDA devices itself and uses
``torch.multiprocessing.spawn`` to launch one worker per GPU, initialising the
same NCCL process group by hand (via an explicit ``tcp://`` rendezvous instead
of the env-var one torchrun sets up). Everything downstream — config, model,
metrics, checkpointing, and the ``<output_dir>/ddp`` output layout — is
unchanged from the original script.

Run it exactly like any other Python script, from anywhere:

    python claude_code/train_eunis_multihead_ddp_selflaunch.py

It always reads ``codex/config.yaml`` (the same configuration file used by the
other entry points) and writes artifacts to ``<config.output_dir>/ddp``, i.e.
the same location the torchrun-launched script uses.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm.auto import tqdm

# codex/eunis_pipeline is a plain, non-installed package living next to the
# original DDP script. Make it importable no matter where this file lives or
# what the current working directory is.
_CODEX_DIR = Path(__file__).resolve().parent.parent / "codex"
if str(_CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEX_DIR))

from eunis_pipeline.config import load_pipeline_config
from eunis_pipeline.data import (
    EunisMultiLabelDataset,
    observed_class_counts,
    read_metadata,
    split_train_validation,
)
from eunis_pipeline.engine import EpochResult
from eunis_pipeline.metrics import (
    soft_confusion_matrix,
    soft_multilabel_cross_entropy,
    top1_soft_multilabel_accuracy,
)
from eunis_pipeline.models import MultiHeadEunisClassifier, build_image_encoder, build_gps_encoder
from eunis_pipeline.reporting import save_epoch_metrics, save_test_artifacts
from eunis_pipeline.transforms import make_transforms

CONFIG_PATH = _CODEX_DIR / "config_ddp.yaml"
# A plain tcp:// rendezvous replaces the env-var init method torchrun sets up.
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation samples without DistributedSampler's padding duplicates."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset, self.rank, self.world_size = dataset, rank, world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return (len(self.dataset) + self.world_size - 1 - self.rank) // self.world_size


def rank_zero(rank: int, message: str) -> None:
    """Prevent duplicate console output from the worker processes."""
    if rank == 0:
        print(f"[INFO] {message}", flush=True)


def gather_arrays(value: np.ndarray, world_size: int) -> np.ndarray:
    """Gather uneven NumPy batches from all ranks without padding sample rows."""
    gathered: list[np.ndarray | None] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return np.concatenate([item for item in gathered if item is not None])


def run_epoch_ddp(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None,
                  device: torch.device, levels: tuple[str, ...], class_counts: dict[str, int],
                  level_weights: tuple[float, ...], label_smoothing: float, split_name: str,
                  rank: int, world_size: int) -> EpochResult:
    """Execute a distributed epoch and return globally aggregated metrics on every rank."""
    training = optimizer is not None
    model.train(training)
    targets, probabilities, predictions = ({level: [] for level in levels} for _ in range(3))
    weighted_loss_sum, sample_count = 0.0, 0
    progress = tqdm(loader, desc=f"{split_name} batches", unit="batch", leave=False, disable=rank != 0)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        gps = batch.get("gps")
        gps = gps.to(device, non_blocking=True) if gps is not None else None
        batch_targets = {level: batch["targets"][level].to(device, non_blocking=True) for level in levels}
        with torch.set_grad_enabled(training):
            outputs = model(images, gps)
            loss = sum(weight * soft_multilabel_cross_entropy(outputs[level], batch_targets[level], label_smoothing)
                       for level, weight in zip(levels, level_weights))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = images.size(0)
        weighted_loss_sum += loss.item() * batch_size
        sample_count += batch_size
        if rank == 0:
            progress.set_postfix(loss=f"{weighted_loss_sum / sample_count:.4f}")
        for level in levels:
            probability = outputs[level].softmax(dim=1).detach().cpu().numpy()
            probabilities[level].append(probability)
            predictions[level].append(probability.argmax(axis=1))
            targets[level].append(batch_targets[level].cpu().numpy())

    # Scalars use a fast tensor collective, while variable-length outputs use
    # all_gather_object. Evaluation samplers ensure each validation/test site is
    # included exactly once in the gathered metrics.
    loss_and_count = torch.tensor([weighted_loss_sum, sample_count], device=device, dtype=torch.float64)
    dist.all_reduce(loss_and_count, op=dist.ReduceOp.SUM)
    matrices, accuracy = {}, {}
    for level in levels:
        target = gather_arrays(np.concatenate(targets[level]), world_size)
        probability = gather_arrays(np.concatenate(probabilities[level]), world_size)
        prediction = gather_arrays(np.concatenate(predictions[level]), world_size)
        accuracy[level] = top1_soft_multilabel_accuracy(target, probability)
        matrices[level] = soft_confusion_matrix(target, prediction, class_counts[level])
    return EpochResult(float(loss_and_count[0] / loss_and_count[1]), accuracy, matrices)


def export_predictions_ddp(model: nn.Module, loader: DataLoader, dataset: EunisMultiLabelDataset,
                           device: torch.device, levels: tuple[str, ...], output_path: Path,
                           rank: int, world_size: int) -> None:
    """Gather rank-local prediction records and write one complete CSV from rank 0."""
    model.eval()
    local_records: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            gps = batch.get("gps")
            outputs = model(batch["image"].to(device, non_blocking=True), gps.to(device, non_blocking=True) if gps is not None else None)
            for index, site_id in enumerate(batch["site_id"]):
                record = {"id_floraveg": int(site_id)}
                for level in levels:
                    probabilities = outputs[level][index].softmax(0)
                    prediction = int(probabilities.argmax())
                    record[f"pred_label_encoded_lvl{level}"] = prediction
                    record[f"pred_label_lvl{level}"] = dataset.label_to_code[level].get(prediction, "<unknown>")
                    record[f"pred_confidence_lvl{level}"] = float(probabilities[prediction])
                    record[f"valid_prediction_lvl{level}"] = bool(batch["targets"][level][index, prediction])
                local_records.append(record)
    gathered: list[list[dict] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_records)
    if rank == 0:
        records = [record for rank_records in gathered if rank_records is not None for record in rank_records]
        metadata = dataset.frame.drop_duplicates(subset=[dataset.data_config.site_id_column])
        metadata.merge(pd.DataFrame(records), how="left", left_on=dataset.data_config.site_id_column, right_on="id_floraveg").to_csv(output_path, index=False)


def worker(local_rank: int, world_size: int) -> None:
    """Per-process entry point invoked by ``torch.multiprocessing.spawn``.

    Same-node only: one process per GPU, so ``rank == local_rank`` throughout.
    """
    rank = local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        rank=rank,
        world_size=world_size,
    )
    try:
        config, runtime = load_pipeline_config(CONFIG_PATH)
        levels, output_dir = config.training.levels, config.output_dir  # / "ddp"
        rank_zero(rank, f"DDP initialised (self-spawned): world_size={world_size}, per-GPU batch={config.training.batch_size}, effective global batch={config.training.batch_size * world_size}")
        rank_zero(rank, f"Train CSV: {config.data.train_csv}; test CSV: {config.data.test_csv}")
        rank_zero(rank, f"Model: img_backbone={config.model.name_img}; gps_backbone={config.model.name_gps}; EUNIS levels: {', '.join(levels)}")
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text(json.dumps(asdict(config), default=str, indent=2))
        dist.barrier()

        train_frame = read_metadata(config.data.train_csv, levels, config.data)
        test_frame = read_metadata(config.data.test_csv, levels, config.data)
        rank_zero(rank, f"Observed train classes: {observed_class_counts(train_frame, levels)}")
        rank_zero(rank, f"Observed test classes: {observed_class_counts(test_frame, levels)}")
        train_frame, validation_frame = split_train_validation(train_frame, levels[-1], config.data.validation_fraction, config.data.split_seed)
        train_tf, evaluation_tf = make_transforms(config.model.name_img)
        datasets = [EunisMultiLabelDataset(frame, config.data.image_dir, levels, config.training.class_counts, transform, config.data)
                    for frame, transform in ((train_frame, train_tf), (validation_frame, evaluation_tf), (test_frame, evaluation_tf))]
        train_sampler = DistributedSampler(datasets[0], num_replicas=world_size, rank=rank, shuffle=True, seed=config.data.split_seed)
        validation_sampler = DistributedEvalSampler(datasets[1], rank, world_size)
        test_sampler = DistributedEvalSampler(datasets[2], rank, world_size)
        loaders = [DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler, num_workers=config.training.num_workers,
                              pin_memory=True, persistent_workers=config.training.num_workers > 0)
                   for dataset, sampler in zip(datasets, (train_sampler, validation_sampler, test_sampler))]

        rank_zero(rank, "Building model and wrapping it with DistributedDataParallel")
        img_encoder = build_image_encoder(config.model.name_img, config.model.pretrained)
        gps_encoder = build_gps_encoder(config.model.name_gps, config.model.gps_embedding_dim)
        base_model = MultiHeadEunisClassifier(img_encoder, gps_encoder, config.model.name_img, config.model.name_gps,
                                              config.training.class_counts, levels,
                                              config.data.use_gps, config.model.fusion,
                                              config.model.freeze_img_backbone, config.model.freeze_gps_backbone).to(device)
        model = DistributedDataParallel(base_model, device_ids=[local_rank], output_device=local_rank)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        best_path, last_path = output_dir / "best_model.pt", output_dir / "last_model.pt"
        start_epoch, best_loss = 0, float("inf")
        if runtime.resume:
            if rank == 0:
                rank_zero(rank, f"Resuming checkpoint: {last_path}")
                checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
            else:
                checkpoint = None
            checkpoint_box = [checkpoint]
            dist.broadcast_object_list(checkpoint_box, src=0)
            checkpoint = checkpoint_box[0]
            model.module.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch, best_loss = checkpoint["epoch"] + 1, checkpoint["best_loss"]

        if not runtime.evaluate_only:
            epochs = tqdm(range(start_epoch, config.training.epochs), desc="DDP epochs", unit="epoch", disable=rank != 0)
            for epoch in epochs:
                train_sampler.set_epoch(epoch)
                train_result = run_epoch_ddp(model, loaders[0], optimizer, device, levels, config.training.class_counts, config.training.level_weights, config.training.label_smoothing, "train", rank, world_size)
                validation_result = run_epoch_ddp(model, loaders[1], None, device, levels, config.training.class_counts, config.training.level_weights, config.training.label_smoothing, "validation", rank, world_size)
                if rank == 0:
                    save_epoch_metrics(output_dir / "train_metrics.csv", epoch, "train", train_result)
                    save_epoch_metrics(output_dir / "validation_metrics.csv", epoch, "validation", validation_result)
                    checkpoint = {"epoch": epoch, "model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "best_loss": best_loss}
                    torch.save(checkpoint, last_path)
                    if validation_result.loss < best_loss:
                        best_loss = validation_result.loss
                        checkpoint["best_loss"] = best_loss
                        torch.save(checkpoint, best_path)
                    rank_zero(rank, f"Epoch {epoch} complete: train_loss={train_result.loss:.4f}, validation_loss={validation_result.loss:.4f}")
                    epochs.set_postfix(train_loss=f"{train_result.loss:.4f}", validation_loss=f"{validation_result.loss:.4f}")
                # Make the new best-loss value identical before the next rank's checkpoint.
                best_tensor = torch.tensor(best_loss, device=device)
                dist.broadcast(best_tensor, src=0)
                best_loss = float(best_tensor)

        # All workers must make the same decision here; raising on rank 0 alone
        # would leave the other ranks blocked at the following collective.
        best_checkpoint_exists = torch.tensor(int(best_path.exists()) if rank == 0 else 0, device=device)
        dist.broadcast(best_checkpoint_exists, src=0)
        if not bool(best_checkpoint_exists):
            raise FileNotFoundError(f"No best checkpoint at {best_path}; train first or enable resume.")
        dist.barrier()
        checkpoint_box = [torch.load(best_path, map_location="cpu", weights_only=False) if rank == 0 else None]
        dist.broadcast_object_list(checkpoint_box, src=0)
        model.module.load_state_dict(checkpoint_box[0]["model"])
        rank_zero(rank, "Running distributed test evaluation")
        test_result = run_epoch_ddp(model, loaders[2], None, device, levels, config.training.class_counts, config.training.level_weights, config.training.label_smoothing, "test", rank, world_size)
        if rank == 0:
            save_test_artifacts(output_dir, test_result)
        export_predictions_ddp(model, loaders[2], datasets[2], device, levels, output_dir / "test_predictions.csv", rank, world_size)
        rank_zero(rank, f"DDP test complete: loss={test_result.loss:.4f}, accuracy={test_result.accuracy}")
    finally:
        dist.destroy_process_group()


def main() -> None:
    """Detect visible GPUs and spawn one DDP worker process per GPU.

    Replaces torchrun's job: it would normally fork one process per
    ``--nproc_per_node`` and export ``RANK``/``LOCAL_RANK``/``WORLD_SIZE``.
    Here, a single ``python`` invocation does both jobs itself.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This DDP entry point is configured for same-node CUDA GPUs. "
            "Use codex/train_eunis_multihead.py for a CPU or single-process run."
        )
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices visible; check CUDA_VISIBLE_DEVICES.")
    print(f"[INFO] Spawning {world_size} DDP worker process(es) locally (no torchrun needed)", flush=True)
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
