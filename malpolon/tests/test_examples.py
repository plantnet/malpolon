"""This script tests the torchgeo datasets module.

**WARNING**: these tests are slow and sometimes require the data to be available.
Thus it is not ran by GitHub workers by default but by a manual trigger of
workflow "dispatch-test-examples".
Furthermore, with the current GitHub CI, tests are ran on CPU and tests
regarding GLC22 are skipped because they require the 60GB of data to be
downloaded manually via Benchmarks/geolifeclef.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

INFO = '\033[93m'
RESET = '\033[0m'
LINK = '\033[94m'
PROJECT_ROOT_PATH = os.getcwd()
TMP_PATHS_TO_DELETE = []
OUT_DIR = "tmp_output"
GPU_ARGS = "trainer.accelerator=gpu trainer.devices=auto" if torch.cuda.is_available() else "trainer.accelerator=cpu trainer.devices=auto"
TRAIN_ARGS = "run.predict=False trainer.max_epochs=2"
INFER_ARGS = f"run.predict_type=test_dataset run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt"
MULTILABEL_ARGS = "task.task=classification_multilabel data.num_classes=5 ~optimizer.metrics 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' '+optimizer.metrics={multilabel_accuracy:{kwargs:{num_labels: ${data.num_classes}}}}'"
MULTICLASS_ARGS = "task.task=classification_multiclass data.num_classes=5 ~optimizer.metrics 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' '+optimizer.metrics={multiclass_accuracy:{kwargs:{num_classes: ${data.num_classes}}}}'"
BINARY_ARGS = "task.task=classification_binary +data.binary_positive_classes=[1] data.num_classes=1 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' ~optimizer.metrics '+optimizer.metrics={binary_accuracy:{kwargs:{}}}'"
EXAMPLE_PATHS = {
    'sentinel-2a-rgbnir': [
        # Multilabel classif
        # Training (raw, transfer learning, inference)
        {"ref": "Train (custom dataset), classification_multilabel, training_raw",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null " + MULTILABEL_ARGS},
        {"ref": "Train (custom dataset), classification_multilabel, training_transfer_learning",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + MULTILABEL_ARGS},
        {"ref": "Train (custom dataset), classification_multilabel, training_inference",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + MULTILABEL_ARGS},
        ## Inference (test_dataset & test_point)
        {"ref": "Inference, classification_multilabel, inference_dataset",
         "path": Path("examples/inference/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} " + MULTILABEL_ARGS},
        {"ref": "Inference, classification_multilabel, inference_point",
         "path": Path("examples/inference/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} run.predict_type=test_point " + MULTILABEL_ARGS},

        # Multiclass classif
        ## Training (raw, transfer learning, inference)
        {"ref": "Train (custom dataset), classification_multiclass, training_raw",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null " + MULTICLASS_ARGS},
        {"ref": "Train (custom dataset), classification_multiclass, training_transfer_learning",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + MULTICLASS_ARGS},
        {"ref": "Train (custom dataset), classification_multiclass, training_inference",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + MULTICLASS_ARGS},
        ## Inference (test_dataset & test_point)
        {"ref": "Inference, classification_multiclass, inference_dataset",
         "path": Path("examples/inference/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} " + MULTICLASS_ARGS},
        {"ref": "Inference, classification_multiclass, inference_point",
         "path": Path("examples/inference/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} run.predict_type=test_point " + MULTICLASS_ARGS},

        # Binary classif
        ## Training (raw, transfer learning, inference)
        {"ref": "Train (custom dataset), classification_binary, training_raw",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null " + BINARY_ARGS},
        {"ref": "Train (custom dataset), classification_binary, training_transfer_learning",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + BINARY_ARGS},
        {"ref": "Train (custom dataset), classification_binary, training_inference",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + BINARY_ARGS},
        ## Inference (test_dataset & test_point)
        {"ref": "Inference, classification_binary, inference_dataset",
         "path": Path("examples/inference/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} " + BINARY_ARGS},
        {"ref": "Inference, classification_binary, inference_point",
         "path": Path("examples/inference/sentinel-2a-rgbnir/cnn_on_rgbnir_torchgeo.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} run.predict_type=test_point " + BINARY_ARGS},
    ],
    'sentinel-2a-rgbnir_bioclim': [
        # Multilabel classif
        # Training (raw, transfer learning, inference)
        {"ref": "Train (custom dataset), classification_multilabel, training_raw",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null " + MULTILABEL_ARGS},
        {"ref": "Train (custom dataset), classification_multilabel, training_transfer_learning",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + MULTILABEL_ARGS},
        {"ref": "Train (custom dataset), classification_multilabel, training_inference",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + MULTILABEL_ARGS},
        ## Inference (test_dataset & test_point)
        {"ref": "Inference, classification_multilabel, inference_dataset",
         "path": Path("examples/inference/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} " + MULTILABEL_ARGS},
        {"ref": "Inference, classification_multilabel, inference_point",
         "path": Path("examples/inference/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} run.predict_type=test_point " + MULTILABEL_ARGS},
        # Multiclass classif
        ## Training (raw, transfer learning, inference)
        {"ref": "Train (custom dataset), classification_multiclass, training_raw",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null " + MULTICLASS_ARGS},
        {"ref": "Train (custom dataset), classification_multiclass, training_transfer_learning",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + MULTICLASS_ARGS},
        {"ref": "Train (custom dataset), classification_multiclass, training_inference",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + MULTICLASS_ARGS},
        ## Inference (test_dataset & test_point)
        {"ref": "Inference, classification_multiclass, inference_dataset",
         "path": Path("examples/inference/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} " + MULTICLASS_ARGS},
        {"ref": "Inference, classification_multiclass, inference_point",
         "path": Path("examples/inference/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} run.predict_type=test_point " + MULTICLASS_ARGS},
        # Binary classif
        ## Training (raw, transfer learning, inference)
        {"ref": "Train (custom dataset), classification_binary, training_raw",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null " + BINARY_ARGS},
        {"ref": "Train (custom dataset), classification_binary, training_transfer_learning",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + BINARY_ARGS},
        {"ref": "Train (custom dataset), classification_binary, training_inference",
         "path": Path("examples/custom_train/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + BINARY_ARGS},
        ## Inference (test_dataset & test_point)
        {"ref": "Inference, classification_binary, inference_dataset",
         "path": Path("examples/inference/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} " + BINARY_ARGS},
        {"ref": "Inference, classification_binary, inference_point",
         "path": Path("examples/inference/sentinel-2a-rgbnir_bioclim/cnn_on_rgbnir_concat.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} run.predict_type=test_point " + BINARY_ARGS},],
    "micro_geolifeclef2022": [
        # Multiclass classif
        ## Training (raw, transfer learning, inference)
        {"ref": "Train (custom dataset), classification_multiclass, training_raw",
         "path": Path("examples/custom_train/micro_geolifeclef2022/cnn_on_rgb_nir_patches.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null"},
        {"ref": "Train (custom dataset), classification_multiclass, training_transfer_learning",
         "path": Path("examples/custom_train/micro_geolifeclef2022/cnn_on_rgb_nir_patches.py"),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt"},
        {"ref": "Train (custom dataset), classification_multiclass, training_inference",
         "path": Path("examples/custom_train/micro_geolifeclef2022/cnn_on_rgb_nir_patches.py"),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt"},
        ## Inference (test_dataset & test_point)
        {"ref": "Inference, classification_multiclass, inference_dataset",
         "path": Path("examples/inference/micro_geolifeclef2022/cnn_on_rgb_nir_patches.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS}"},
        {"ref": "Inference, classification_multiclass, inference_point",
         "path": Path("examples/inference/micro_geolifeclef2022/cnn_on_rgb_nir_patches.py"),
         "hydra_args": f"{GPU_ARGS} {INFER_ARGS} run.predict_type=test_point"},],
}

GLC22_EXAMPLE_PATHS = {
    # GeolifeCLEF 2022 training are really long and the data folder heavy, so they are commented by default
    "geolifeclef2022": [
        {"ref": "Benchmarks/geolifeclef, classification_multiclass, training_raw",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2022/cnn_on_rgb_patches.py'),
         "hydra_args": f"{GPU_ARGS} trainer.max_epochs=2"},
        {"ref": "Benchmarks/geolifeclef, classification_multiclass, training_raw",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2022/cnn_on_rgb_temperature_patches.py'),
         "hydra_args": f"{GPU_ARGS} trainer.max_epochs=2"},
        {"ref": "Benchmarks/geolifeclef, classification_multiclass, training_raw",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2022/cnn_on_temperature_patches.py'),
         "hydra_args": f"{GPU_ARGS} trainer.max_epochs=2"},
    ],
}

GLC23_EXAMPLE_PATHS = {
    "geolifeclef2023": [
        # Multilabel classif
        ## Training (raw, transfer learning, inference)
        {"ref": "Benchmarks/geolifeclef, classification_multilabel, training_raw",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2023/cnn_on_rgbnir_glc23_patches.py'),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null " + "data.labels_name=dataset/sample_data/Presence_only_occurrences/Presences_only_train_sample_mutilabel_dummy.csv task.task=classification_multilabel data.num_classes=96 ~optimizer.metrics 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' '+optimizer.metrics={multilabel_accuracy:{kwargs:{num_labels: ${data.num_classes}}}}'"},
        {"ref": "Benchmarks/geolifeclef, classification_multilabel, training_transfer_learning",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2023/cnn_on_rgbnir_glc23_patches.py'),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + "data.labels_name=dataset/sample_data/Presence_only_occurrences/Presences_only_train_sample_mutilabel_dummy.csv task.task=classification_multilabel data.num_classes=96 ~optimizer.metrics 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' '+optimizer.metrics={multilabel_accuracy:{kwargs:{num_labels: ${data.num_classes}}}}'"},
        {"ref": "Benchmarks/geolifeclef, classification_multilabel, training_inference",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2023/cnn_on_rgbnir_glc23_patches.py'),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + "data.labels_name=dataset/sample_data/Presence_only_occurrences/Presences_only_train_sample_mutilabel_dummy.csv task.task=classification_multilabel data.num_classes=96 ~optimizer.metrics 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' '+optimizer.metrics={multilabel_accuracy:{kwargs:{num_labels: ${data.num_classes}}}}'"},

        # Multiclass classif
        ## Training (raw, transfer learning, inference)
        {"ref": "Benchmarks/geolifeclef, classification_multiclass, training_raw",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2023/cnn_on_rgbnir_glc23_patches.py'),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null " + "data.labels_name=dataset/sample_data/Presence_only_occurrences/Presences_only_train_sample.csv task.task=classification_multiclass data.num_classes=9827 ~optimizer.metrics 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' '+optimizer.metrics={multiclass_accuracy:{kwargs:{num_classes: ${data.num_classes}}}}'"},
        {"ref": "Benchmarks/geolifeclef, classification_multiclass, training_transfer_learning",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2023/cnn_on_rgbnir_glc23_patches.py'),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + "data.labels_name=dataset/sample_data/Presence_only_occurrences/Presences_only_train_sample.csv task.task=classification_multiclass data.num_classes=9827 ~optimizer.metrics 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' '+optimizer.metrics={multiclass_accuracy:{kwargs:{num_classes: ${data.num_classes}}}}'"},
        {"ref": "Benchmarks/geolifeclef, classification_multiclass, training_inference",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2023/cnn_on_rgbnir_glc23_patches.py'),
         "hydra_args": f"{GPU_ARGS} run.predict=True run.checkpoint_path={OUT_DIR}_training_raw/last.ckpt " + "data.labels_name=dataset/sample_data/Presence_only_occurrences/Presences_only_train_sample.csv task.task=classification_multiclass data.num_classes=9827 ~optimizer.metrics 'model.modifiers.change_last_layer.num_outputs=${data.num_classes}' '+optimizer.metrics={multiclass_accuracy:{kwargs:{num_classes: ${data.num_classes}}}}'"},

        # Data loading examples
        {"ref": "Benchmarks/geolifeclef, patch, data_loading",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2023/example_patch_loading.py'),
         "hydra_args": ""},
        {"ref": "Benchmarks/geolifeclef, time_series, data_loading",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2023/example_time_series_loading.py'),
         "hydra_args": ""},
    ],
}

GLC24_PRE_EXTRACTED_EXAMPLE_PATHS = {
    "geolifeclef2024_pre_extracted": [
        # Multilabel classif (species)
        ## Training (raw, transfer learning, inference)
        {"ref": "Benchmarks/geolifeclef/geolifeclef2024_pre_extracted, classification_multilabel (species), training_raw",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2024_pre_extracted/glc24_cnn_multimodal_ensemble.py'),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null ~trainer.val_check_interval loggers.exp_name=glc24_pre_extracted_mme_test"},

        # Multiclass classif (habitats)
        ## Training (raw, transfer learning, inference)
        {"ref": "Benchmarks/geolifeclef/geolifeclef2024_pre_extracted, classification_multiclass (habitats), training_raw",
         "path": Path('examples/benchmarks/geolifeclef/geolifeclef2024_pre_extracted/glc24_cnn_multimodal_ensemble_habitat.py'),
         "hydra_args": f"{GPU_ARGS} {TRAIN_ARGS} run.checkpoint_path=null ~trainer.val_check_interval loggers.exp_name=glc24_pre_extracted_mme_test"},
    ],
}

# @pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
def test_train_inference_examples():
    ckpt_path = ''
    for expe_name, v in EXAMPLE_PATHS.items():
        print(f'\n{INFO}[INFO] --- Scenarios "Train (custom datasets)" / "inference" --- {RESET}')
        print(f'\n{INFO}[INFO] Testing example: {expe_name}{RESET}{INFO}...{RESET}')
        for expes in v:
            ref, path, args = expes['ref'], expes['path'], expes['hydra_args']
            expe_type = ref.rsplit(', ', maxsplit=1)[-1].lower()
            print(f'{INFO}[INFO]   > {LINK}{path.name}{RESET}{INFO}: {ref}...{RESET}\n')
            assert path.exists()
            os.chdir(path.parent)

            out_dir = Path(f"{OUT_DIR}_{ref.rsplit(' ', maxsplit=1)[-1]}")
            if out_dir.exists():
                os.system(f'rm -rf {out_dir}')
            if any(v in expe_type for v in ['training_raw', 'training_transfer_learning']):
                a = os.system(f"python {path.name} {args} hydra.run.dir={out_dir}")  # 5-6x faster than subprocess.run or popen
                assert not a
                if expe_type != 'training_transfer_learning':
                    assert os.path.isfile(out_dir / 'last.ckpt')  # When using transfer learning, last.ckpt is not guaranteed to exist as lightning my overwrite it with the same link referencing itself and breaking if there are no "proper" checkpoints to reference (which is the case when begining the transfer learning task)
                    ckpt_path = Path(os.getcwd()) / out_dir
                assert os.path.isfile(out_dir / 'metrics.csv')
                assert os.path.isfile(out_dir / f'{path.stem}.log')
                assert os.path.isfile(out_dir / 'hparams.yaml')
                assert os.path.isdir(out_dir / 'tensorboard_logs')
            elif 'inference' in expe_type:
                a = os.system(f'python {path.name} hydra.run.dir={out_dir} {args} run.checkpoint_path={ckpt_path}/last.ckpt')
                assert not a
                if expe_type != 'inference_dataset':
                    assert os.path.isfile(out_dir / 'prediction_point.csv')
                if expe_type != 'inference_point':
                    assert os.path.isfile(out_dir / 'predictions_test_dataset.csv')

            TMP_PATHS_TO_DELETE.append(Path(os.getcwd()) / out_dir)
            os.chdir(PROJECT_ROOT_PATH)
            print(f'\n{INFO}[INFO] OK. {RESET}')

    # Clean up: remove the output files
    print(f'\n{INFO}[INFO] Cleaning up temporary test output files... {RESET}')
    for path in TMP_PATHS_TO_DELETE:
        os.system(f'rm -rf {path}')
        print(f'{INFO}         > {LINK}{path}{RESET}')
    print(f'\n{INFO}[INFO] Done. {RESET}')


# @pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
def test_GLC22_examples():
    ckpt_path = ''
    for expe_name, v in GLC22_EXAMPLE_PATHS.items():
        print(f'\n{INFO}[INFO] --- Scenarios "benchmarks/geolifeclef" --- {RESET}')
        print(f'\n{INFO}[INFO] Testing example: {expe_name}{RESET}{INFO}...{RESET}')
        for expes in v:
            ref, path, args = expes['ref'], expes['path'], expes['hydra_args']
            expe_type = ref.rsplit(', ', maxsplit=1)[-1].lower()
            print(f'{INFO}[INFO]   > {LINK}{path.name}{RESET}{INFO}: {ref}...{RESET}\n')
            assert path.exists()
            os.chdir(path.parent)

            out_dir = Path(f"{OUT_DIR}_{ref.rsplit(' ', maxsplit=1)[-1]}")
            if out_dir.exists():
                os.system(f'rm -rf {out_dir}')
            if any(v in expe_type for v in ['training_raw', 'training_transfer_learning']):
                a = os.system(f"python {path.name} {args} hydra.run.dir={out_dir}")  # 5-6x faster than subprocess.run or popen
                assert not a
                if expe_type != 'training_transfer_learning':
                    assert os.path.isfile(out_dir / 'last.ckpt')  # When using transfer learning, last.ckpt is not guaranteed to exist as lightning my overwrite it with the same link referencing itself and breaking if there are no "proper" checkpoints to reference (which is the case when begining the transfer learning task)
                    ckpt_path = Path(os.getcwd()) / out_dir
                assert os.path.isfile(out_dir / 'metrics.csv')
                assert os.path.isfile(out_dir / f'{path.stem}.log')
                assert os.path.isfile(out_dir / 'hparams.yaml')
                assert os.path.isdir(out_dir / 'tensorboard_logs')
            elif 'inference' in expe_type:
                a = os.system(f'python {path.name} hydra.run.dir={out_dir} {args} run.checkpoint_path={ckpt_path}/last.ckpt')
                assert not a
                if expe_type != 'inference_dataset':
                    assert os.path.isfile(out_dir / 'prediction_point.csv')
                if expe_type != 'inference_point':
                    assert os.path.isfile(out_dir / 'predictions_test_dataset.csv')
            elif 'data_loading' in expe_type:
                a = os.system(f"python {path.name}")
                assert not a

            TMP_PATHS_TO_DELETE.append(Path(os.getcwd()) / out_dir)
            os.chdir(PROJECT_ROOT_PATH)
            print(f'\n{INFO}[INFO] OK. {RESET}')

    # Clean up: remove the output files
    print(f'\n{INFO}[INFO] Cleaning up temporary test output files... {RESET}')
    for path in TMP_PATHS_TO_DELETE:
        os.system(f'rm -rf {path}')
        print(f'{INFO}         > {LINK}{path}{RESET}')
    print(f'\n{INFO}[INFO] Done. {RESET}')


# @pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
def test_GLC23_examples():
    ckpt_path = ''
    for expe_name, v in GLC23_EXAMPLE_PATHS.items():
        print(f'\n{INFO}[INFO] --- Scenarios "benchmarks/geolifeclef" --- {RESET}')
        print(f'\n{INFO}[INFO] Testing example: {expe_name}{RESET}{INFO}...{RESET}')
        for expes in v:
            ref, path, args = expes['ref'], expes['path'], expes['hydra_args']
            expe_type = ref.rsplit(', ', maxsplit=1)[-1].lower()
            print(f'{INFO}[INFO]   > {LINK}{path.name}{RESET}{INFO}: {ref}...{RESET}\n')
            assert path.exists()
            os.chdir(path.parent)

            out_dir = Path(f"{OUT_DIR}_{ref.rsplit(' ', maxsplit=1)[-1]}")
            if out_dir.exists():
                os.system(f'rm -rf {out_dir}')
            if any(v in expe_type for v in ['training_raw', 'training_transfer_learning']):
                a = os.system(f"python {path.name} {args} hydra.run.dir={out_dir}")  # 5-6x faster than subprocess.run or popen
                assert not a
                if expe_type != 'training_transfer_learning':
                    assert os.path.isfile(out_dir / 'last.ckpt')  # When using transfer learning, last.ckpt is not guaranteed to exist as lightning my overwrite it with the same link referencing itself and breaking if there are no "proper" checkpoints to reference (which is the case when begining the transfer learning task)
                    ckpt_path = Path(os.getcwd()) / out_dir
                assert os.path.isfile(out_dir / 'metrics.csv')
                assert os.path.isfile(out_dir / f'{path.stem}.log')
                assert os.path.isfile(out_dir / 'hparams.yaml')
                assert os.path.isdir(out_dir / 'tensorboard_logs')
            elif 'inference' in expe_type:
                a = os.system(f'python {path.name} hydra.run.dir={out_dir} {args} run.checkpoint_path={ckpt_path}/last.ckpt')
                assert not a
                if expe_type != 'inference_dataset':
                    assert os.path.isfile(out_dir / 'prediction_point.csv')
                if expe_type != 'inference_point':
                    assert os.path.isfile(out_dir / 'predictions_test_dataset.csv')
            elif 'data_loading' in expe_type:
                a = os.system(f"python {path.name}")
                assert not a

            TMP_PATHS_TO_DELETE.append(Path(os.getcwd()) / out_dir)
            os.chdir(PROJECT_ROOT_PATH)
            print(f'\n{INFO}[INFO] OK. {RESET}')

    # Clean up: remove the output files
    print(f'\n{INFO}[INFO] Cleaning up temporary test output files... {RESET}')
    for path in TMP_PATHS_TO_DELETE:
        os.system(f'rm -rf {path}')
        print(f'{INFO}         > {LINK}{path}{RESET}')
    print(f'\n{INFO}[INFO] Done. {RESET}')


@pytest.mark.skip(reason="Impossible for pytest to run because user input is needed to validate data download.")
def test_GLC24_pre_extracted_examples():
    ckpt_path = ''
    for expe_name, v in GLC24_PRE_EXTRACTED_EXAMPLE_PATHS.items():
        print(f'\n{INFO}[INFO] --- Scenarios "benchmarks/geolifeclef2024_pre_etracted" --- {RESET}')
        print(f'\n{INFO}[INFO] Testing example: {expe_name}{RESET}{INFO}...{RESET}')
        for expes in v:
            ref, path, args = expes['ref'], expes['path'], expes['hydra_args']
            expe_type = ref.rsplit(', ', maxsplit=1)[-1].lower()
            print(f'{INFO}[INFO]   > {LINK}{path.name}{RESET}{INFO}: {ref}...{RESET}\n')
            assert path.exists()
            os.chdir(path.parent)

            out_dir = Path(f"{OUT_DIR}_{ref.rsplit(' ', maxsplit=1)[-1]}")
            if out_dir.exists():
                os.system(f'rm -rf {out_dir}')

            # Create a temporary lightweight observation file
            if 'habitat' in str(path):
                df = pd.read_csv('dataset/geolifeclef-2024_habitats/GLC24_PA_metadata_habitats-lvl3_train_split-10.0%_val.csv').sample(n=100)
            else:
                df = pd.read_csv('dataset/geolifeclef-2024/GLC24_PA_metadata_train_val-10.0min.csv').sample(n=100)
            df.to_csv('obs_sample.csv', index=False, sep=',')
            TMP_PATHS_TO_DELETE.append(Path(os.getcwd()) / 'obs_sample.csv')
            args += " 'data.metadata_paths.train=obs_sample.csv' 'data.metadata_paths.val=obs_sample.csv' 'data.metadata_paths.test=obs_sample.csv' "

            if any(v in expe_type for v in ['training_raw', 'training_transfer_learning']):
                a = os.system(f"python {path.name} {args} hydra.run.dir={out_dir} ")  # 5-6x faster than subprocess.run or popen
                assert not a
                if expe_type != 'training_transfer_learning':
                    assert os.path.isfile(out_dir / 'last.ckpt')  # When using transfer learning, last.ckpt is not guaranteed to exist as lightning my overwrite it with the same link referencing itself and breaking if there are no "proper" checkpoints to reference (which is the case when begining the transfer learning task)
                    ckpt_path = Path(os.getcwd()) / out_dir
                assert os.path.isfile(out_dir / 'glc24_pre_extracted_mme_test/metrics.csv')
                assert os.path.isfile(out_dir / 'glc24_pre_extracted_mme_test/hparams.yaml')
                assert os.path.isfile(out_dir / f'{path.stem}.log')
                assert os.path.isdir(out_dir / 'tensorboard_logs')
            elif 'inference' in expe_type:
                a = os.system(f'python {path.name} hydra.run.dir={out_dir} {args} run.checkpoint_path={ckpt_path}/last.ckpt')
                assert not a
                if expe_type != 'inference_dataset':
                    assert os.path.isfile(out_dir / 'prediction_point.csv')
                if expe_type != 'inference_point':
                    assert os.path.isfile(out_dir / 'predictions_test_dataset.csv')
            elif 'data_loading' in expe_type:
                a = os.system(f"python {path.name}")
                assert not a

            TMP_PATHS_TO_DELETE.append(Path(os.getcwd()) / out_dir)
            os.chdir(PROJECT_ROOT_PATH)
            print(f'\n{INFO}[INFO] OK. {RESET}')

    # Clean up: remove the output files
    print(f'\n{INFO}[INFO] Cleaning up temporary test output files... {RESET}')
    for path in TMP_PATHS_TO_DELETE:
        os.system(f'rm -rf {path}')
        print(f'{INFO}         > {LINK}{path}{RESET}')
    print(f'\n{INFO}[INFO] Done. {RESET}')
