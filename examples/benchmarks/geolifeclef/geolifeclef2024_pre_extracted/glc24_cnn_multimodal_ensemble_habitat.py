"""Main script to run training or inference on GLC24 pre_extracted dataset.

This script runs the GeoLifeCLEF2024 pre-extracted dataset to predict
habitats distribution using the Multi-Modal Ensemble model (MME).

Author: Theo Larcher <theo.larcher@inria.fr>
"""

import logging

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from malpolon.data.datasets.geolifeclef2024_pre_extracted import \
    GLC24DatamoduleHabitats
from malpolon.logging import Summary
from malpolon.models.custom_models.glc2024_pre_extracted_prediction_system import \
    ClassificationSystemGLC24


def set_seed(seed):
    import lightning.pytorch as pl
    from lightning.pytorch import seed_everything

    # Set seed for Python's built-in random number generator
    torch.manual_seed(seed)
    # Set seed for numpy
    np.random.seed(seed)
    seed_everything(seed, workers=True)
    # Set seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set cuDNN's random number generator seed for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@hydra.main(version_base="1.3", config_path="config/", config_name="glc24_cnn_multimodal_ensemble_habitat")
def main(cfg: DictConfig):
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """
    set_seed(69)
    # Loggers
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log_dir = log_dir.split(hydra.utils.get_original_cwd())[1][1:]  # Transforming absolute path to relative path
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version=cfg.loggers.exp_name)
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.loggers.log_dir_name, version=cfg.loggers.exp_name)
    logger_tb.log_hyperparams(cfg)
    logger = logging.getLogger("lightning.pytorch.core")
    logger.addHandler(logging.FileHandler(f"{log_dir}/core.log"))

    # Datamodule & Model
    datamodule = GLC24DatamoduleHabitats(**cfg.data, **cfg.task)
    classif_system = ClassificationSystemGLC24(cfg.model, **cfg.optim, **cfg.task,
                                               checkpoint_path=cfg.run.checkpoint_path,
                                               weights_dir=log_dir,
                                               num_classes=cfg.data.num_classes)  # multiclass

    # Lightning Trainer
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="checkpoint-{epoch:02d}-{step}-{loss/val:.4f}",
            monitor="loss/val",
            mode="min",
            save_on_train_epoch_end=True,
            save_last=True,
            every_n_train_steps=75,
        ),
    ]
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer, deterministic=True)

    # Run
    if cfg.run.predict:
        model_loaded = ClassificationSystemGLC24.load_from_checkpoint(classif_system.checkpoint_path,
                                                                      model=classif_system.model,
                                                                      hparams_preprocess=False,
                                                                      strict=False,
                                                                      weights_dir=log_dir)

        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           np.arange(cfg.data.num_classes))
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='predictions_test_dataset', top_k=None, return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:1])

    else:
        trainer.fit(classif_system, datamodule=datamodule, ckpt_path=classif_system.checkpoint_path)
        trainer.validate(classif_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
