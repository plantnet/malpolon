import numpy as np

import torch
from torchvision import transforms


class Occ_Lat_Long_Transform:
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        data = transforms.Resize(256)(data)
        return data

class Full_True_Clean_Subset_Transform:
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        return data
            
class RGBDataTransform:
    def __call__(self, data):
        return transforms.functional.to_tensor(data)
    
    
class TCI_Sentinel_Transform:
    def __call__(self, data):
        data = transforms.functional.to_tensor(data) 
        data = transforms.CenterCrop(size=300)(data)
        data = transforms.Resize(256)(data)
        return data 

class Baty_95m_Transform :                                                  
    def __call__(self, data):
        #data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(31)(data))) :
            mean_data = np.nanmean(transforms.CenterCrop(31)(data))
        elif not np.isnan(np.nanmean(transforms.CenterCrop(51)(data))) :
            mean_data = np.nanmean(transforms.CenterCrop(51)(data))
        else :
            mean_data = np.nanmean(data)
        np.nan_to_num(data, copy=False, nan=np.nanmean(mean_data))   
        data = transforms.CenterCrop(31)(data)
        data = transforms.Resize(256)(data)
        return data

class Bathymetry_Transform :                                                  
    def __call__(self, data):
        #data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)  
        data = transforms.CenterCrop(20)(data)
        data = transforms.Resize(256)(data)
        return data

class Meditereanean_Sst_Transform :                                                  
    def __call__(self, data):
        #data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(32)(data))) :
            mean_data = np.nanmean(transforms.CenterCrop(32)(data))
        elif not np.isnan(np.nanmean(transforms.CenterCrop(50)(data))) :
            mean_data = np.nanmean(transforms.CenterCrop(50)(data))
        else :
            mean_data = np.nanmean(data)
        np.nan_to_num(data, copy=False, nan=np.nanmean(mean_data))   
        data = transforms.CenterCrop(32)(data)
        data = transforms.Resize(256)(data)
        return data


class Chlorophyll_Concentration_1km_Transform :                                                  
    def __call__(self, data):
        #data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(30)(data))) :
            mean_data = np.nanmean(transforms.CenterCrop(30)(data))
        elif not np.isnan(np.nanmean(transforms.CenterCrop(50)(data))) :
            mean_data = np.nanmean(transforms.CenterCrop(50)(data))
        else :
            mean_data = np.nanmean(data)
        np.nan_to_num(data, copy=False, nan=np.nanmean(mean_data))   
        data = transforms.CenterCrop(30)(data)
        data = transforms.Resize(256)(data)
        return data


class Standard_3_Bands_16_To_14_Pixels_Transform :                                                  
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(15)(data[0:,:,]))) :
            mean_data_0 = np.nanmean(transforms.CenterCrop(15)(data[0:,:,]))
        else :
            mean_data_0 = np.nanmean(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(15)(data[1:,:,]))) :
            mean_data_1 = np.nanmean(transforms.CenterCrop(15)(data[1:,:,]))
        else :
            mean_data_1 = np.nanmean(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(15)(data[2:,:,]))) :
            mean_data_2 = np.nanmean(transforms.CenterCrop(15)(data[2:,:,]))
        else :
            mean_data_2 = np.nanmean(data)
        np.nan_to_num(data[0:,:,], copy=False, nan=np.nanmean(mean_data_0))
        np.nan_to_num(data[1:,:,], copy=False, nan=np.nanmean(mean_data_1))   
        np.nan_to_num(data[2:,:,], copy=False, nan=np.nanmean(mean_data_2))      
        data = transforms.CenterCrop(15)(data)
        data = transforms.Resize(256)(data)
        return data
    
class Standard_3_Bands_30_To_14_Pixels_Transform :                                                  
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(15)(data[0:,:,]))) :
            mean_data_0 = np.nanmean(transforms.CenterCrop(15)(data[0:,:,]))
        elif not np.isnan(np.nanmean(transforms.CenterCrop(20)(data[0:,:,]))) :
            mean_data_0 = np.nanmean(transforms.CenterCrop(20)(data[0:,:,]))
        else :
            mean_data_0 = np.nanmean(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(15)(data[1:,:,]))) :
            mean_data_1 = np.nanmean(transforms.CenterCrop(15)(data[1:,:,]))
        elif not np.isnan(np.nanmean(transforms.CenterCrop(20)(data[1:,:,]))) :
            mean_data_1 = np.nanmean(transforms.CenterCrop(20)(data[1:,:,]))
        else :
            mean_data_1 = np.nanmean(data)
        if not np.isnan(np.nanmean(transforms.CenterCrop(15)(data[2:,:,]))) :
            mean_data_2 = np.nanmean(transforms.CenterCrop(15)(data[2:,:,]))
        elif not np.isnan(np.nanmean(transforms.CenterCrop(20)(data[2:,:,]))) :
            mean_data_2 = np.nanmean(transforms.CenterCrop(20)(data[2:,:,]))
        else :
            mean_data_2 = np.nanmean(data)
        np.nan_to_num(data[0:,:,], copy=False, nan=np.nanmean(mean_data_0))
        np.nan_to_num(data[1:,:,], copy=False, nan=np.nanmean(mean_data_1))   
        np.nan_to_num(data[2:,:,], copy=False, nan=np.nanmean(mean_data_2))      
        data = transforms.CenterCrop(15)(data)
        data = transforms.Resize(256)(data)
        return data
