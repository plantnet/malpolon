# Classe pour effecteur les changement défini dans la partie model.modifiers du .yaml 
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
from tqdm import tqdm
import glob 
from pathlib import Path
from omegaconf import OmegaConf,open_dict

class NewConvolutionalLayerInitFuncStrategy:
    def __init__(self, strategy, num_input_channels, rescaling=False,):
        self.strategy = strategy
        self.rescaling = rescaling
        self.num_input_channels = num_input_channels

    def __call__(self, old_layer, new_layer):
        with torch.no_grad():
            if self.strategy == "random_init":
                new_layer.weight[:, :3] = old_layer.weight
            elif self.strategy == "red_pretraining":
                #new_layer.weight[:] = old_layer.weight[:, [0, 1, 2, 0, 1]]
                #new_layer.weight[:] = old_layer.weight[:,([0,1,2]*math.ceil(self.num_input_channels/3))[:int(math.ceil(self.num_input_channels))]]
                new_layer.weight[:] = torch.cat(
                    (old_layer.weight[:,[0,1,2]],
                    ((old_layer.weight[:, [0]] + old_layer.weight[:, [1]] + old_layer.weight[:, [2]])/3)[:, [0,]*(self.num_input_channels - 3)]),
                    dim=1)
            elif self.strategy == "red_pretraining_mean":
                new_layer.weight[:] = ((old_layer.weight[:, [0]] + old_layer.weight[:, [1]] + old_layer.weight[:, [2]])/3)[:, [0,]*self.num_input_channels]
            
            if self.rescaling:
                new_layer.weight *= 3 / 4

            if hasattr(new_layer, "bias"):
                new_layer.bias = old_layer.bias
                

# pour l'utilisation de la loss ImbalNoisedTopK, il faut normaliser le poid et modiffier leur propagation 
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out
       


from dataset import load_patch       

def Init_of_secondary_parameters(cfg):
    print()
    print("Récupération des informations suivantes :\n- nombre d'entrées pour le modèle (si num_input_channels=auto)\n- nombre de sorties pour le modèle (si num_outputs=auto) \n- des suffixes des pataches par variables (.tif, jp2, etc.) \n- nombre d'occurence par classe")   
    print()
    
    # liste du nombre d'occurence par classe (trier de la classe 0 à +inf) 
    df = pd.read_csv(cfg.data.csv_occurence_path, sep=cfg.data.csv_separator, index_col=cfg.data.csv_col_occurence_id)  
    df_train = df[df.subset=='train']
    cls_num_list_train = df_train[cfg.data.csv_col_class_id].value_counts().sort_index().tolist()
    
    # liste des extentions pour chaque variables
    patch_data_ext=[]
    for var in tqdm(cfg.patch.patch_data) :
        filename=glob.glob(cfg.data.dataset_path + '/**/' + str(df.index[0]) + '_' + var + '*')[0]
        patch_data_ext.append(Path(filename).suffix)
    
    # calcule du nombre de classe et intégration du resultat dans cfg
    if cfg.model.modifiers.change_last_layer.num_outputs == 'auto':
        n_classes = len(df[cfg.data.csv_col_class_id].unique())
        cfg.model.modifiers.change_last_layer.num_outputs = int(n_classes)        
    
    # calcule du nombre de bande en entrée et intégration du resultat dans cfg
    if cfg.model.modifiers.change_first_convolutional_layer.num_input_channels == 'auto':
        id_init = df[df.subset=='train'].index[0]
        patches = load_patch(id_init, cfg.data.dataset_path + "/patches", data=cfg.patch.patch_data, patch_data_ext = patch_data_ext)
        patch_band = []
        for var in cfg.patch.patch_data:
            if len(patches[var].shape) == 3:
                for band in range(0,patches[var].shape[2]):
                    patch_band.append(var + '_' + str(band))
            elif len(patches[var].shape) == 2:
                patch_band.append(var + '_0')
        cfg.model.modifiers.change_first_convolutional_layer.num_input_channels = int(len(patch_band))      
    
    # Créé un cfg modifier pour prendre en comptre le num_input_channels pour la classe NewConvolutionalLayerInitFuncStrategy 
    # qui récupère en paramettre toute les valeurs dans de la branche "new_conv_layer_init_func" de cfg / cfg_modif.
    # -> ce n'est pas le plus propre mais ça évite de modifier le coeur du code de Malpolon
    cfg_modif=cfg.copy()
    OmegaConf.set_struct(cfg_modif, True)
    with open_dict(cfg_modif):
        cfg_modif.model.modifiers.change_first_convolutional_layer.new_conv_layer_init_func.num_input_channels = int(cfg.model.modifiers.change_first_convolutional_layer.num_input_channels)        


    return cls_num_list_train, patch_data_ext, cfg, cfg_modif



