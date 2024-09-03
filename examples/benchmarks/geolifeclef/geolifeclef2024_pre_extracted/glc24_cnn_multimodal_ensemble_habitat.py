import logging
import os
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
from torchvision.datasets.utils import download_and_extract_archive

from malpolon.data.datasets.geolifeclef2024_pre_extracted import (
    GLC24Datamodule, TestDataset, TrainDataset)
from malpolon.logging import Summary
from malpolon.models.custom_models.glc2024_multimodal_ensemble_model import \
    MultimodalEnsemble
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


class TrainDatasetHabitat(TrainDataset):
    def __init__(self, metadata, classes, bioclim_data_dir=None, landsat_data_dir=None, sentinel_data_dir=None, transform=None, task='classification_multilabel'):
        metadata = metadata[metadata['habitatId'].notna()]
        metadata = metadata[metadata['habitatId'] != 'Unknown']
        self.label_encoder = LabelEncoder().fit(classes)
        metadata['habitatId_encoded'] = self.label_encoder.transform(metadata['habitatId'])
        metadata.rename({'PlotObservationID': 'surveyId'}, axis=1, inplace=True)
        metadata.rename({'habitatId_encoded': 'speciesId'}, axis=1, inplace=True)

        super().__init__(metadata, num_classes=len(classes), bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform, task=task)

        self.metadata = metadata
        if 'speciesId' in self.metadata.columns:
            self.metadata = self.metadata.dropna(subset='speciesId').reset_index(drop=True)
            self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        else:
            self.metadata['speciesId'] = [None] * len(self.metadata)
        self.metadata = self.metadata.drop_duplicates(subset=['surveyId', 'habitatId']).reset_index(drop=True)  # Should we ?
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        self.targets = self.metadata['speciesId'].values
        self.observation_ids = self.metadata['surveyId']

class TestDatasetHabitat(TestDataset):
    def __init__(self, metadata, classes, bioclim_data_dir=None, landsat_data_dir=None, sentinel_data_dir=None, transform=None, task='classification_multilabel'):
        metadata = metadata[metadata['habitatId'].notna()]
        metadata = metadata[metadata['habitatId'] != 'Unknown']
        self.label_encoder = LabelEncoder().fit(classes)
        metadata['habitatId_encoded'] = self.label_encoder.transform(metadata['habitatId'])
        metadata.rename({'PlotObservationID': 'surveyId'}, axis=1, inplace=True)
        metadata.rename({'habitatId_encoded': 'speciesId'}, axis=1, inplace=True)

        super().__init__(metadata, bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform, task=task)

        self.metadata = self.metadata.drop_duplicates(subset=['surveyId', 'habitatId']).reset_index(drop=True)  # Should we ?
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        self.targets = self.metadata['speciesId'].values
        self.observation_ids = self.metadata['surveyId']


class GLC24DatamoduleHabitats(GLC24Datamodule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Union of train and test cls
        HABITAT_CLS = ['MA221', 'MA222', 'MA223', 'MA224', 'MA225', 'MA241', 'MA251',
                       'MA252', 'MA253', 'MAa', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16',
                       'N18', 'N19', 'N1A', 'N1B', 'N1D', 'N1G', 'N1H', 'N1J', 'N21',
                       'N22', 'N31', 'N32', 'N35', 'Q11', 'Q21', 'Q22', 'Q24', 'Q25',
                       'Q41', 'Q42', 'Q43', 'Q44', 'Q51', 'Q52', 'Q53', 'Q54', 'Q61',
                       'Q62', 'Q63', 'R11', 'R12', 'R13', 'R14', 'R16', 'R18', 'R19',
                       'R1A', 'R1B', 'R1D', 'R1E', 'R1F', 'R1H', 'R1M', 'R1P', 'R1Q',
                       'R1R', 'R1S', 'R21', 'R22', 'R23', 'R24', 'R31', 'R32', 'R34',
                       'R35', 'R36', 'R37', 'R41', 'R43', 'R44', 'R45', 'R51', 'R52',
                       'R54', 'R55', 'R56', 'R57', 'R61', 'R62', 'R63', 'S21', 'S22',
                       'S23', 'S24', 'S25', 'S26', 'S31', 'S32', 'S33', 'S34', 'S35',
                       'S36', 'S37', 'S38', 'S41', 'S42', 'S51', 'S52', 'S53', 'S54',
                       'S61', 'S62', 'S63', 'S91', 'S92', 'S93', 'T11', 'T12', 'T13',
                       'T15', 'T16', 'T17', 'T18', 'T19', 'T1A', 'T1B', 'T1C', 'T1D',
                       'T1E', 'T1F', 'T1H', 'T21', 'T22', 'T24', 'T27', 'T29', 'T31',
                       'T32', 'T33', 'T34', 'T35', 'T36', 'T37', 'T39', 'T3A', 'T3C',
                       'T3D', 'T3F', 'T3J', 'T3K', 'T3M', 'U22', 'U24', 'U26', 'U27',
                       'U28', 'U29', 'U32', 'U33', 'U34', 'U36', 'U37', 'U38', 'U3A',
                       'U3D', 'U71', 'V11', 'V12', 'V13', 'V14', 'V15', 'V32', 'V33',
                       'V34', 'V35', 'V37', 'V38', 'V39']
        self.classes = HABITAT_CLS

    def get_dataset(self, split, transform, **kwargs):
        match split:
            case 'train':
                train_metadata = pd.read_csv(self.metadata_paths['train'])
                dataset = TrainDatasetHabitat(train_metadata, self.classes, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_train = dataset
            case 'val':
                val_metadata = pd.read_csv(self.metadata_paths['val'])
                dataset = TrainDatasetHabitat(val_metadata,  self.classes, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_val = dataset
            case 'test':
                test_metadata = pd.read_csv(self.metadata_paths['test'])
                dataset = TestDatasetHabitat(test_metadata, self.classes, **self.data_paths['test'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_test = dataset
        return dataset

    def _check_integrity_habitat(self):
        paths = {'predictors': ['EnvironmentalRasters', 'PA-test-landsat_time_series',
                                'PA_Test_SatellitePatches_NIR', 'PA_Test_SatellitePatches_RGB',
                                'PA-train-landsat_time_series', 'PA_Train_SatellitePatches_NIR',
                                'PA_Train_SatellitePatches_RGB', 'TimeSeries-Cubes'],
                 'metadata': ['GLC24_PA_metadata_habitats-lvl3_test.csv',
                              'GLC24_PA_metadata_habitats-lvl3_train.csv',
                              'GLC24_PA_metadata_habitats-lvl3_train_split-10.0%_all.csv',
                              'GLC24_PA_metadata_habitats-lvl3_train_split-10.0%_train.csv',
                              'GLC24_PA_metadata_habitats-lvl3_train_split-10.0%_val.csv']}
        downloaded_p = all(map(lambda x: Path(self.root / x).exists(), paths['predictors']))
        downloaded_m = all(map(lambda x: Path(self.root / x).exists(), paths['metadata']))
        return downloaded_p, downloaded_m

    def download(self):
        downloaded_p, downloaded_m = self._check_integrity_habitat()
        if not downloaded_p:
            print('Downloading data ("predictors")...')
            self.root = self.root.parent / "geolifeclef-2024"
            super().download()
            self.root = self.root.parent / "geolifeclef-2024_habitats"
            links = {"../geolifeclef-2024/TimeSeries-Cubes/": "TimeSeries-Cubes",
                     "../geolifeclef-2024/PA_Train_SatellitePatches_RGB/": "PA_Train_SatellitePatches_RGB",
                     "../geolifeclef-2024/PA_Train_SatellitePatches_NIR/": "PA_Train_SatellitePatches_NIR",
                     "../geolifeclef-2024/PA-train-landsat_time_series/": "PA-train-landsat_time_series",
                     "../geolifeclef-2024/PA_Test_SatellitePatches_RGB/": "PA_Test_SatellitePatches_RGB",
                     "../geolifeclef-2024/PA_Test_SatellitePatches_NIR/": "PA_Test_SatellitePatches_NIR",
                     "../geolifeclef-2024/PA-test-landsat_time_series/": "PA-test-landsat_time_series",
                     "../geolifeclef-2024/EnvironmentalRasters/": "EnvironmentalRasters"}
            for k, v in links.items():
                os.system(f'ln -sf {k} {str(self.root / v)}')
        else:
            print('Data ("predictors") already downloaded.')

        if not downloaded_m:
            print('Downloading observations ("metadata")...')
            download_and_extract_archive(
                "https://lab.plantnet.org/seafile/f/583b1878f0694eeca163/?dl=1",
                self.root,
                filename='GLC24_PA_metadata_habitats-lvl3.zip',
                md5='24dc7e126f2bac79a63fdacb4f210f19',
                remove_finished=True
            )
        else:
            print('Observations ("metadata") already downloaded.')


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
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version=cfg.loggers.exp_name)
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.loggers.log_dir_name, version=cfg.loggers.exp_name)
    logger_tb.log_hyperparams(cfg)
    logger = logging.getLogger("lightning.pytorch.core")
    logger.addHandler(logging.FileHandler(f"{log_dir}/core.log"))

    # Datamodule & Model
    datamodule = GLC24DatamoduleHabitats(**cfg.data, **cfg.task)
    classif_system = ClassificationSystemGLC24(cfg.model, **cfg.optimizer, **cfg.task,
                                               checkpoint_path=cfg.run.checkpoint_path,
                                               weights_dir=log_dir,
                                               num_classes=cfg.data.num_classes)  # multiclass

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
            every_n_train_steps=75,
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
                                                           np.arange(cfg.data.num_classes))
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='predictions_test_dataset', top_k=None, return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:1])

    else:
        trainer.fit(classif_system, datamodule=datamodule, ckpt_path=classif_system.checkpoint_path)
        trainer.validate(classif_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
