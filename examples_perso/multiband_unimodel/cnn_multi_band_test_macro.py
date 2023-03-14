import os


import hydra
from omegaconf import DictConfig

import torch
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms
from typing import Mapping, Union

from malpolon.data.data_module import BaseDataModule
from malpolon.models.standard_prediction_systems import GenericPredictionSystemLrScheduler
from malpolon.models.utils import check_model
from malpolon.logging import Summary

from dataset import MicroGeoLifeCLEF2022Dataset
from transforms import *
from auto_plot import Autoplot
from pytopk import BalNoisedTopK
from pytopk import ImbalNoisedTopK

from init_elements import Init_of_secondary_parameters

from pytorch_lightning.callbacks import LearningRateMonitor
from auto_lr_finder import Auto_lr_find
import torchmetrics.functional as Fmetrics


class PreprocessData():
    def __init__(self, patch_band_mean, patch_band_sd):
        self.patch_band_mean = patch_band_mean
        self.patch_band_sd = patch_band_sd

    def __call__(self, data ):
        if "full_true_clean_subset" in list(data.keys()) : 
            full_true_clean_subset_data = data["full_true_clean_subset"]
            full_true_clean_subset_data = Full_True_Clean_Subset_Transform()(full_true_clean_subset_data)
            full_true_clean_subset_data = transforms.Normalize(self.patch_band_mean["full_true_clean_subset"], self.patch_band_sd["full_true_clean_subset"])(full_true_clean_subset_data)

        if "bathymetry" in list(data.keys()) : 
            bathymetry_data = data["bathymetry"]
            bathymetry_data = Bathymetry_Transform()(bathymetry_data)
            bathymetry_data = transforms.Normalize(self.patch_band_mean["bathymetry"], self.patch_band_sd["bathymetry"])(bathymetry_data)

        if "bathy_95m" in list(data.keys()) : 
            bathy_95m_data = data["bathy_95m"]
            bathy_95m_data = Baty_95m_Transform()(bathy_95m_data)
            bathy_95m_data = transforms.Normalize(self.patch_band_mean["bathy_95m"], self.patch_band_sd["bathy_95m"])(bathy_95m_data)

        if "chlorophyll_concentration_1km" in list(data.keys()) : 
            chlorophyll_concentration_1km_data = data["chlorophyll_concentration_1km"]
            chlorophyll_concentration_1km_data = Chlorophyll_Concentration_1km_Transform()(chlorophyll_concentration_1km_data)
            chlorophyll_concentration_1km_data = transforms.Normalize(self.patch_band_mean["chlorophyll_concentration_1km"], self.patch_band_sd["chlorophyll_concentration_1km"])(chlorophyll_concentration_1km_data)

        if "east_water_velocity_4_2km_mean_day_lite" in list(data.keys()) : 
            east_water_velocity_4_2km_mean_day_lite_data = data["east_water_velocity_4_2km_mean_day_lite"]
            east_water_velocity_4_2km_mean_day_lite_data = Standard_3_Bands_16_To_14_Pixels_Transform()(east_water_velocity_4_2km_mean_day_lite_data)
            east_water_velocity_4_2km_mean_day_lite_data = transforms.Normalize(self.patch_band_mean["east_water_velocity_4_2km_mean_day_lite"], self.patch_band_sd["east_water_velocity_4_2km_mean_day_lite"])(east_water_velocity_4_2km_mean_day_lite_data)

        if "east_water_velocity_4_2km_mean_month_lite" in list(data.keys()) : 
            east_water_velocity_4_2km_mean_month_lite_data = data["east_water_velocity_4_2km_mean_month_lite"]
            east_water_velocity_4_2km_mean_month_lite_data = Standard_3_Bands_16_To_14_Pixels_Transform()(east_water_velocity_4_2km_mean_month_lite_data)
            east_water_velocity_4_2km_mean_month_lite_data = transforms.Normalize(self.patch_band_mean["east_water_velocity_4_2km_mean_month_lite"], self.patch_band_sd["east_water_velocity_4_2km_mean_month_lite"])(east_water_velocity_4_2km_mean_month_lite_data)

        if "east_water_velocity_4_2km_mean_year_lite" in list(data.keys()) : 
            east_water_velocity_4_2km_mean_year_lite_data = data["east_water_velocity_4_2km_mean_year_lite"]
            east_water_velocity_4_2km_mean_year_lite_data = Standard_3_Bands_16_To_14_Pixels_Transform()(east_water_velocity_4_2km_mean_year_lite_data)
            east_water_velocity_4_2km_mean_year_lite_data = transforms.Normalize(self.patch_band_mean["east_water_velocity_4_2km_mean_year_lite"], self.patch_band_sd["east_water_velocity_4_2km_mean_year_lite"])(east_water_velocity_4_2km_mean_year_lite_data)
        
        if "meditereanean_sst" in list(data.keys()) : 
            meditereanean_sst_data = data["meditereanean_sst"]
            meditereanean_sst_data = Meditereanean_Sst_Transform()(meditereanean_sst_data)
            meditereanean_sst_data = transforms.Normalize(self.patch_band_mean["meditereanean_sst"], self.patch_band_sd["meditereanean_sst"])(meditereanean_sst_data)

        if "north_water_velocity_4_2km_mean_day_lite" in list(data.keys()) : 
            north_water_velocity_4_2km_mean_day_lite_data = data["north_water_velocity_4_2km_mean_day_lite"]
            north_water_velocity_4_2km_mean_day_lite_data = Standard_3_Bands_16_To_14_Pixels_Transform()(north_water_velocity_4_2km_mean_day_lite_data)
            north_water_velocity_4_2km_mean_day_lite_data = transforms.Normalize(self.patch_band_mean["north_water_velocity_4_2km_mean_day_lite"], self.patch_band_sd["north_water_velocity_4_2km_mean_day_lite"])(north_water_velocity_4_2km_mean_day_lite_data)

        if "north_water_velocity_4_2km_mean_month_lite" in list(data.keys()) : 
            north_water_velocity_4_2km_mean_month_lite_data = data["north_water_velocity_4_2km_mean_month_lite"]
            north_water_velocity_4_2km_mean_month_lite_data = Standard_3_Bands_16_To_14_Pixels_Transform()(north_water_velocity_4_2km_mean_month_lite_data)
            north_water_velocity_4_2km_mean_month_lite_data = transforms.Normalize(self.patch_band_mean["north_water_velocity_4_2km_mean_month_lite"], self.patch_band_sd["north_water_velocity_4_2km_mean_month_lite"])(north_water_velocity_4_2km_mean_month_lite_data)

        if "north_water_velocity_4_2km_mean_year_lite" in list(data.keys()) : 
            north_water_velocity_4_2km_mean_year_lite_data = data["north_water_velocity_4_2km_mean_year_lite"]
            north_water_velocity_4_2km_mean_year_lite_data = Standard_3_Bands_16_To_14_Pixels_Transform()(north_water_velocity_4_2km_mean_year_lite_data)
            north_water_velocity_4_2km_mean_year_lite_data = transforms.Normalize(self.patch_band_mean["north_water_velocity_4_2km_mean_year_lite"], self.patch_band_sd["north_water_velocity_4_2km_mean_year_lite"])(north_water_velocity_4_2km_mean_year_lite_data)

        if "occ_lat_long" in list(data.keys()) : 
            occ_lat_long_data = data["occ_lat_long"]
            occ_lat_long_data = Occ_Lat_Long_Transform()(occ_lat_long_data)
            occ_lat_long_data = transforms.Normalize(self.patch_band_mean["occ_lat_long"], self.patch_band_sd["occ_lat_long"])(occ_lat_long_data)

        if "salinity_4_2km_mean_day_lite" in list(data.keys()) : 
            salinity_4_2km_mean_day_lite_data = data["salinity_4_2km_mean_day_lite"]
            salinity_4_2km_mean_day_lite_data = Standard_3_Bands_30_To_14_Pixels_Transform()(salinity_4_2km_mean_day_lite_data)
            salinity_4_2km_mean_day_lite_data = transforms.Normalize(self.patch_band_mean["salinity_4_2km_mean_day_lite"], self.patch_band_sd["salinity_4_2km_mean_day_lite"])(salinity_4_2km_mean_day_lite_data)

        if "salinity_4_2km_mean_month_lite" in list(data.keys()) : 
            salinity_4_2km_mean_month_lite_data = data["salinity_4_2km_mean_month_lite"]
            salinity_4_2km_mean_month_lite_data = Standard_3_Bands_30_To_14_Pixels_Transform()(salinity_4_2km_mean_month_lite_data)
            salinity_4_2km_mean_month_lite_data = transforms.Normalize(self.patch_band_mean["salinity_4_2km_mean_month_lite"], self.patch_band_sd["salinity_4_2km_mean_month_lite"])(salinity_4_2km_mean_month_lite_data)

        if "salinity_4_2km_mean_year_lite" in list(data.keys()) : 
            salinity_4_2km_mean_year_lite_data = data["salinity_4_2km_mean_year_lite"]
            salinity_4_2km_mean_year_lite_data = Standard_3_Bands_30_To_14_Pixels_Transform()(salinity_4_2km_mean_year_lite_data)
            salinity_4_2km_mean_year_lite_data = transforms.Normalize(self.patch_band_mean["salinity_4_2km_mean_year_lite"], self.patch_band_sd["salinity_4_2km_mean_year_lite"])(salinity_4_2km_mean_year_lite_data)
        
        if "sea_water_potential_temperature_at_sea_floor_4_2km_mean_day" in list(data.keys()) : 
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_data = data["sea_water_potential_temperature_at_sea_floor_4_2km_mean_day"]
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_data = Standard_3_Bands_16_To_14_Pixels_Transform()(sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_data)
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_data = transforms.Normalize(self.patch_band_mean["sea_water_potential_temperature_at_sea_floor_4_2km_mean_day"], self.patch_band_sd["sea_water_potential_temperature_at_sea_floor_4_2km_mean_day"])(sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_data)

        if "sea_water_potential_temperature_at_sea_floor_4_2km_mean_month" in list(data.keys()) : 
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_data = data["sea_water_potential_temperature_at_sea_floor_4_2km_mean_month"]
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_data = Standard_3_Bands_16_To_14_Pixels_Transform()(sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_data)
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_data = transforms.Normalize(self.patch_band_mean["sea_water_potential_temperature_at_sea_floor_4_2km_mean_month"], self.patch_band_sd["sea_water_potential_temperature_at_sea_floor_4_2km_mean_month"])(sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_data)

        if "sea_water_potential_temperature_at_sea_floor_4_2km_mean_year" in list(data.keys()) : 
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_year_data = data["sea_water_potential_temperature_at_sea_floor_4_2km_mean_year"]
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_year_data = Standard_3_Bands_16_To_14_Pixels_Transform()(sea_water_potential_temperature_at_sea_floor_4_2km_mean_year_data)
            sea_water_potential_temperature_at_sea_floor_4_2km_mean_year_data = transforms.Normalize(self.patch_band_mean["sea_water_potential_temperature_at_sea_floor_4_2km_mean_year"], self.patch_band_sd["sea_water_potential_temperature_at_sea_floor_4_2km_mean_year"])(sea_water_potential_temperature_at_sea_floor_4_2km_mean_year_data)

        if "TCI_sentinel" in list(data.keys()) : 
            TCI_sentinel_data = data["TCI_sentinel"]
            TCI_sentinel_data = TCI_Sentinel_Transform()(TCI_sentinel_data)
            TCI_sentinel_data = transforms.Normalize(self.patch_band_mean["TCI_sentinel"], self.patch_band_sd["TCI_sentinel"])(TCI_sentinel_data)

        str_patch_data = ""
        for patch_data in list(data.keys()):
            str_patch_data += patch_data + "_data, "

        return eval("torch.concat((" + str_patch_data + "))") 


class MicroGeoLifeCLEF2022DataModule(BaseDataModule):
    """
    Data module for MicroGeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        csv_occurence_path: str,
        csv_separator:str,
        csv_col_occurence_id:str,
        csv_col_class_id: str,
        train_batch_size: int,
        inference_batch_size: int,
        num_workers: int,
        patch_data_ext: list,
        patch_data: list,
        patch_band_mean: dict,
        patch_band_sd: dict,
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.csv_occurence_path = csv_occurence_path
        self.csv_separator = csv_separator
        self.csv_col_occurence_id = csv_col_occurence_id
        self.patch_data_ext = patch_data_ext
        self.patch_data = patch_data
        self.patch_band_mean = patch_band_mean
        self.patch_band_sd = patch_band_sd

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                PreprocessData(self.patch_band_mean, self.patch_band_sd),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                PreprocessData(self.patch_band_mean, self.patch_band_sd),
            ]
        )

    def prepare_data(self):
        MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            csv_separator = self.csv_separator,
            csv_col_occurence_id = self.csv_col_occurence_id,
            patch_data_ext = self.patch_data_ext,
            subset = "train",
            use_rasters=False,
            csv_occurence_path = self.csv_occurence_path 
        )

    def get_dataset(self, split, transform, **kwargs):
        dataset = MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            csv_separator = self.csv_separator,
            csv_col_occurence_id = self.csv_col_occurence_id,
            patch_data_ext = self.patch_data_ext,
            subset = split,
            patch_data=self.patch_data,
            use_rasters=False,
            csv_occurence_path = self.csv_occurence_path,
            transform=transform,
            **kwargs
        )
        return dataset

class ClassificationSystem(GenericPredictionSystemLrScheduler):
    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        mode: str,
        factor: float,
        patience: int,
        threshold: float, 
        cooldown: int,
        logging_interval: str,
        metric_to_track: str, 
        loss_type: str,
        k: int,
        epsilon: float,
        max_m: float,
        cls_num_list_train: list):

        num_outputs = model.modifiers.change_last_layer.num_outputs
        model = check_model(model)
                 
        if loss_type == 'BalNoisedTopK':
            loss =  BalNoisedTopK(k=k, epsilon=epsilon)
        elif loss_type == 'ImbalNoisedTopK':
            from init_elements import NormedLinear
            model.fc = NormedLinear(model.fc.in_features, model.fc.out_features)
            loss =  ImbalNoisedTopK(k=k, epsilon=epsilon, max_m=max_m, cls_num_list=cls_num_list_train)
        else :
            loss = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,)
        
        scheduler = {'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, cooldown=cooldown, threshold=threshold),
                     'metric_to_track': metric_to_track}
    
        metrics = {
            
            "accuracy_v0": Fmetrics.accuracy,
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=1).to(device = "cuda"),
            "accuracy_macro_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=1, num_classes=num_outputs, average="macro"),
            "accuracy_macro":  torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=1).to(device = "cuda"),
            
            "top_5_accuracy_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=5),
            "top_5_accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=5).to(device = "cuda"),
            "top_5_accuracy_macro_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=5, num_classes=num_outputs, average="macro"),
            "top_5_accuracy_macro":  torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=5).to(device = "cuda"),
            
            "top_10_accuracy_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=10),
            "top_10_accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=10).to(device = "cuda"),
            "top_10_accuracy_macro_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=10, num_classes=num_outputs, average="macro"),
            "top_10_accuracy_macro":  torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=10).to(device = "cuda"),

            "top_20_accuracy_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=20),
            "top_10_accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=10).to(device = "cuda"),
            "top_20_accuracy_macro_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=30, num_classes=num_outputs, average="macro"),
            "top_20_accuracy_macro":  torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=30).to(device = "cuda"),
            
            "top_30_accuracy_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=30),
            "top_30_accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=30).to(device = "cuda"),
            "top_30_accuracy_macro_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=30, num_classes=num_outputs, average="macro"),
            "top_30_accuracy_macro":  torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=30).to(device = "cuda"),

            "top_150_accuracy_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=150),
            "top_150_accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=150).to(device = "cuda"),
            "top_150_accuracy_macro_v0": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=150, num_classes=num_outputs, average="macro"),
            "top_150_accuracy_macro":  torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=150).to(device = "cuda"),
            }
            
        super().__init__(model, loss, optimizer, scheduler, metrics)






@hydra.main(version_base="1.1", config_path="config", config_name="cnn_multi_band_config")
def main(cfg: DictConfig) -> None:
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)
    
    cls_num_list_train, patch_data_ext, cfg, cfg_modif = Init_of_secondary_parameters(cfg=cfg)

    datamodule = MicroGeoLifeCLEF2022DataModule(**cfg.data,
                                                patch_data_ext = patch_data_ext,
                                                patch_data=cfg.patch.patch_data, 
                                                patch_band_mean = cfg.patch.patch_band_mean,
                                                patch_band_sd = cfg.patch.patch_band_sd)

    if cfg.visualization.check_dataloader != True :
        cfg_model = hydra.utils.instantiate(cfg_modif.model)
        

        if cfg.visualization.auto_lr_finder == True :
            Auto_lr_find(cfg, cfg_model, datamodule, cls_num_list_train)

        else : 
            model = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, cls_num_list_train=cls_num_list_train)

            callbacks = [
                Summary(),
                ModelCheckpoint(
                    dirpath=os.getcwd(),
                    filename="checkpoint-{epoch:02d}-{step}-{val_accuracy:.4f}",
                    monitor= cfg.callbacks.monitor,
                    mode=cfg.callbacks.mode,),
                LearningRateMonitor(logging_interval=cfg.optimizer.scheduler.logging_interval), #epoch'),
                EarlyStopping(monitor=cfg.callbacks.monitor, min_delta=0.00, patience=cfg.callbacks.patience, verbose=False, mode=cfg.callbacks.mode),
            ]                
            
            trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

            trainer.fit(model, datamodule=datamodule)   # pour charger un model et continuer l'entrainement : trainer.fit(..., ckpt_path="some/path/to/my_checkpoint.ckpt")

            trainer.validate(model, datamodule=datamodule)

            Autoplot(os.getcwd(), cfg.visualization.graph)
    
    else : 
        from check_dataloader import Check_dataloader
        Check_dataloader(datamodule, cfg, patch_data_ext)
 

if __name__ == "__main__":
    main()