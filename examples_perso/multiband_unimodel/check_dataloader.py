import os
import shutil
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset import load_patch

def Check_dataloader(datamodule, cfg, patch_data_ext):
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    classes=batch[1].numpy().tolist()

    df = pd.read_csv(cfg.data.csv_occurence_path, sep=cfg.data.csv_separator, index_col=cfg.data.csv_col_occurence_id)  
    id_init = df[df.subset=='train'].index[0]
    patches = load_patch(id_init, cfg.data.dataset_path + "/patches", data=cfg.patch.patch_data, patch_data_ext = patch_data_ext)
    patch_band = []
    for var in cfg.patch.patch_data:
        if len(patches[var].shape) == 3:
            for band in range(0,patches[var].shape[2]):
                patch_band.append(var + '_' + str(band))
        elif len(patches[var].shape) == 2:
            patch_band.append(var + '_0')

    if not os.path.exists(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'check_dataloader/'):
        os.makedirs(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'check_dataloader/')
    
    fig = plt.figure(figsize=(10, 60)) #5*nombre de ligne 
    rows = len(classes)
    columns = 3
    for j in range(0, len(patch_band)) :
        print(f"Création de l'image de check_dataloader {j+1}/{len(patch_band)}.")    
        for i in tqdm(range(0, 30)) :
            fig.add_subplot(rows, columns, i+1)
            patch=batch[0][i:,j,:,:]
            patch=patch.numpy()
            plt.imshow(patch[0])
            plt.axis('off')
            plt.title("class : " + str(classes[i]))
        fig.savefig(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'check_dataloader/' + patch_band[j] + '.png' )
    
    shutil.rmtree(os.getcwd())