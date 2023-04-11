import os

import hydra
from omegaconf import DictConfig


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from malpolon.logging import Summary

from transforms import *
from auto_plot import Autoplot

from init_elements import Init_of_secondary_parameters

from pytorch_lightning.callbacks import LearningRateMonitor

from transfer_learning import Transfer_learning_ia_biodiv
from auto_lr_finder import Auto_lr_find
from datamodule import MicroGeoLifeCLEF2022DataModule, ClassificationSystem


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
        
    if cfg.visualization.check_dataloader == True :   
        from check_dataloader import Check_dataloader
        Check_dataloader(datamodule, cfg, patch_data_ext)
    
    elif cfg.visualization.auto_lr_finder == True :
        cfg_model = hydra.utils.instantiate(cfg_modif.model)
        Auto_lr_find(cfg, cfg_model, datamodule, cls_num_list_train)

    else : 
        cfg_model = hydra.utils.instantiate(cfg_modif.model)
        model = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, cls_num_list_train=cls_num_list_train)

        callbacks = [
            Summary(),
            ModelCheckpoint(
                dirpath=os.getcwd(),
                filename="checkpoint-{epoch:02d}-{step}-{val_accuracy:.4f}",
                monitor= cfg.callbacks.monitor,
                mode=cfg.callbacks.mode,),
            LearningRateMonitor(logging_interval=cfg.optimizer.scheduler.logging_interval), #epoch'),
            EarlyStopping(monitor=cfg.callbacks.monitor, min_delta=0.00, patience=cfg.callbacks.patience, verbose=False, mode=cfg.callbacks.mode)]                
            
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)




        transfer_learning = True 
        if transfer_learning == True :
            model, datamodule, trainer = Transfer_learning_ia_biodiv(model, cfg, cfg_model, cls_num_list_train, patch_data_ext, logger)

            


        trainer.fit(model, datamodule=datamodule)   # pour charger un model et continuer l'entrainement : trainer.fit(..., ckpt_path="some/path/to/my_checkpoint.ckpt")
            
        trainer.validate(model, datamodule=datamodule)

        Autoplot(os.getcwd(), cfg.visualization.graph)


if __name__ == "__main__":
    main()