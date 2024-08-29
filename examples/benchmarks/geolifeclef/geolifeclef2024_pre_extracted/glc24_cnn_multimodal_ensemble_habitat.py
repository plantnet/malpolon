import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

from malpolon.data.datasets.geolifeclef2024_pre_extracted import (
    GLC24Datamodule, TestDataset, TrainDataset)
from malpolon.logging import Summary
from malpolon.models.custom_models.glc2024_multimodal_ensemble_model import \
    MultimodalEnsemble
from malpolon.models.custom_models.glc2024_pre_extracted_prediction_system import \
    ClassificationSystemGLC24

HABITAT_CLS = ['N1G', 'R1A', 'Q22', 'Q24', 'S41', 'T21', 'MA225', 'R35', 'Q52',
               'R1D', 'Q51', 'R56', 'T19', 'V32', 'R36', 'V35', 'S42', 'R18',
               'R22', 'N14', 'T17', 'N1H', 'T16', 'MA222', 'V38', 'R37', 'R55',
               'V37', 'T3A', 'N16', 'MA223', 'T24', 'R43', 'N1D', 'S51', 'R1P',
               'T1B', 'MA224', 'R1F', 'T15', 'R21', 'V11', 'R1E', 'N15', 'R1R',
               'U27', 'R1B', 'S35', 'S32', 'T3M', 'S92', 'T1H', 'T3J', 'T11',
               'N1A', 'Q53', 'R1Q', 'V34', 'R63', 'MA252', 'Q42', 'T1E', 'Q25',
               'V33', 'N12', 'T31', 'S53', 'T13', 'R1M', 'R51', 'T27', 'S38',
               'S21', 'S31', 'V15', 'T33', 'R23', 'S26', 'U26', 'R19', 'R1H',
               'N1B', 'S61', 'T1C', 'R13', 'N32', 'MA253', 'Q21', 'T22', 'S33',
               'T1F', 'Q11', 'R57', 'S34', 'U22', 'R1S', 'R44', 'V39', 'T29',
               'T36', 'Q54', 'T1D', 'T12', 'U38', 'R41', 'R16', 'S91', 'T37',
               'T32', 'S22', 'T18', 'U28', 'R31', 'R32', 'S24', 'MA241', 'N19',
               'N22', 'T3D', 'U36', 'R62', 'T35', 'N35', 'S25', 'N11', 'N18',
               'S54', 'R54', 'T1A', 'MA221', 'V13', 'R45', 'Q43', 'U37', 'S36',
               'S52', 'N31', 'S62', 'N1J', 'S37', 'S23', 'R52', 'T39', 'R14',
               'R24', 'Q44', 'R34', 'MA251', 'Q41', 'R61', 'R12', 'U3D', 'N21',
               'S63', 'U33', 'T34', 'T3F', 'U34', 'V14', 'T3C', 'S93', 'T3K',
               'V12', 'U32', 'U3A', 'U29', 'R11', 'S74']


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


class TrainDatasetHabitat(TrainDataset):
    def __init__(self, metadata, num_classes=168, bioclim_data_dir=None, landsat_data_dir=None, sentinel_data_dir=None, transform=None, task='classification_multilabel'):
        metadata = metadata[metadata['habitatId'].notna()]
        metadata = metadata[metadata['habitatId'] != 'Unknown']
        self.label_encoder = LabelEncoder().fit(metadata['habitatId'])
        metadata['habitatId_encoded'] = self.label_encoder.transform(metadata['habitatId'])
        metadata.rename({'PlotObservationID': 'surveyId'}, axis=1, inplace=True)
        metadata.rename({'habitatId_encoded': 'speciesId'}, axis=1, inplace=True)

        super().__init__(metadata, num_classes=num_classes, bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform, task=task)

        self.metadata = metadata
        if 'speciesId' in self.metadata.columns:
            self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
            self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        else:
            self.metadata['speciesId'] = [None] * len(self.metadata)
        self.metadata = self.metadata.drop_duplicates(subset=["surveyId", "habitatId"]).reset_index(drop=True)  # Should we ?
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()


class TestDatasetHabitat(TestDataset, TrainDatasetHabitat):
    def __init__(self, metadata, bioclim_data_dir=None, landsat_data_dir=None, sentinel_data_dir=None, transform=None):
        super().__init__(metadata, bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform)


class GLC24DatamoduleHabitats(GLC24Datamodule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dataset(self, split, transform, **kwargs):
        match split:
            case 'train':
                train_metadata = pd.read_csv(self.metadata_paths['train'])
                dataset = TrainDatasetHabitat(train_metadata, self.num_classes, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_train = dataset
            case 'val':
                val_metadata = pd.read_csv(self.metadata_paths['val'])
                dataset = TrainDatasetHabitat(val_metadata,  self.num_classes, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_val = dataset
            case 'test':
                test_metadata = pd.read_csv(self.metadata_paths['test'])
                dataset = TestDatasetHabitat(test_metadata, **self.data_paths['test'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_test = dataset
        return dataset

    def _check_integrity(self):
        downloaded = (self.root / "eva_header.csv").exists()
        # split = (self.root / "GLC24_PA_metadata_train_train-10.0min.csv").exists()
        # if downloaded and not split:
        #     print('Data already downloaded but not split. Splitting data spatially into train (90%) & val (10%) sets.')
        #     split_obs_spatially(str(self.root / "GLC24_PA_metadata_train.csv"), val_size=0.10)
        #     split = True
        return downloaded  # and split

@hydra.main(version_base="1.3", config_path="config/", config_name="glc24_cnn_multimodal_ensemble_habitat2")
def main(cfg: DictConfig) -> None:
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
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version=cfg.loggers.exp_name)
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.loggers.log_dir_name, version=cfg.loggers.exp_name)
    logger_tb.log_hyperparams(cfg)
    logger = logging.getLogger("lightning.pytorch.core")
    logger.addHandler(logging.FileHandler(f"{log_dir}/core.log"))

    # Datamodule & Model
    datamodule = GLC24DatamoduleHabitats(**cfg.data, **cfg.task)
    classif_system = ClassificationSystemGLC24(cfg.model, **cfg.optimizer,
                                               checkpoint_path=cfg.run.checkpoint_path,
                                               weights_dir=log_dir)  # multilabel

    # Lightning Trainer
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

    # Run
    if cfg.run.predict:
        model_loaded = ClassificationSystemGLC24.load_from_checkpoint(classif_system.checkpoint_path,
                                                                      model=classif_system.model,
                                                                      hparams_preprocess=False,
                                                                      strict=False)

        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           np.arange(cfg.data.num_classes),
                                                           activation_fn=torch.nn.Sigmoid())
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='predictions_test_dataset', top_k=25, return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:1])

    else:
        trainer.fit(classif_system, datamodule=datamodule, ckpt_path=classif_system.checkpoint_path)
        trainer.validate(classif_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
