
import logging

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from malpolon.data.datasets.geolifeclef2024_pre_extracted import \
    GLC24Datamodule
from malpolon.logging import Summary
from malpolon.models.geolifeclef2024_multimodal_ensemble import (
    ClassificationSystemGLC24, MultimodalEnsemble)


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


@hydra.main(version_base="1.3", config_path="config/", config_name="glc24_cnn_multimodal_ensemble")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """
    set_seed(69)
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version=cfg.loggers.exp_name)
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.loggers.log_dir_name, version=cfg.loggers.exp_name)
    logger_tb.log_hyperparams(cfg)

    logger = logging.getLogger("lightning.pytorch.core")
    logger.addHandler(logging.FileHandler(f"{log_dir}/core.log"))

    datamodule = GLC24Datamodule(**cfg.data, **cfg.task)
    model = MultimodalEnsemble(num_classes=cfg.model.modifiers.change_last_layer.num_outputs,
                               positive_weigh_factor=cfg.model.positive_weigh_factor)
    classif_system = ClassificationSystemGLC24(model, **cfg.optimizer)  # multilabel

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="checkpoint-{epoch:02d}-{step}-{" + f"loss/val" + ":.4f}",
            monitor=f"loss/val",
            mode="min",
            save_on_train_epoch_end=True,
            save_last=True,
            every_n_train_steps=100,
        ),
    ]
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer, deterministic=True)

    if cfg.run.predict:
        model_loaded = ClassificationSystemGLC24.load_from_checkpoint(cfg.run.checkpoint_path,
                                                                      model=classif_system.model,
                                                                      hparams_preprocess=False,
                                                                      strict=False)

        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           np.arange(cfg.data.num_classes))
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='predictions_test_dataset', top_k=25, return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:1])

    else:
        trainer.fit(classif_system, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        trainer.validate(classif_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
