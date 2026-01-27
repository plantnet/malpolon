"""This file contains custom functions for training a multiscale JRC model.

Mainly contains custom PyTorch Dataloader methods and sampler.

Author: Theo Larcher <theo.larcher@inria.fr>
"""
import os
import torch
from typing import Optional, Tuple, Union, Iterator, Any, Callable, List
from torch.utils.data import Sampler, Dataset, DataLoader

# 2. Custom collate function returning directly a list of dictionaries with {'img': img_tensor, 'gps': gps_tuple}. But this implies adding a loop over the multi-dimensional tensors which defeats the purpose of batching.
def collate_species(original_batch):
    imgs, gpss, inds, ids = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    inds_batched = torch.stack(list(inds), dim=0)
    ids_batched = torch.cat(ids, dim=0)
    return img_batched, gps_batched, inds_batched, ids_batched

def collate_landscape(original_batch):
    imgs, gpss, inds, ids = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    inds_batched = torch.stack(list(inds), dim=0)
    ids_batched = torch.cat(ids, dim=0)
    return img_batched, gps_batched, inds_batched, ids_batched

# Version multi-view per row
# def collate_landscape(original_batch):
#     imgs, gpss = zip(*original_batch)
#     img_batched = torch.cat(list(imgs), dim=0)  # Reshape to stack the views along the batch dim. Output is: [imgA_view1, imgA_view2, ..., imgB_view1, imgB_view2...]
#     gps_batched = torch.stack(list(gpss), dim=0)
#     # In order to address the inconsistent number of views of LUCAS images, we must choose a strategy between the 2 following:
    
#     # a) Reshaping imgs to stack the views on the channel dim. This requires to adapt the model to accept k channels with k>3 probably.
#     # img_batched = img_batched.reshape(1, -1, img_batched.shape[2], img_batched.shape[3])[0] 
    
#     # b) Repeating the gps embeddings to match the new expanded batch dim because of LUCAS views. This requires to add an if case in the contrastive loss computation as the shapes of the similarity matrix are based on the batch_size which is artificially expanded.
#     repeats = torch.tensor([x.shape[0] for x in imgs])
#     gps_batched = torch.repeat_interleave(gps_batched, repeats, dim=0)  # Output is: [gps_imgA, gps_imgA,..., gps_imgB, gps_imgB...]
#     return img_batched, gps_batched

def collate_satellite(original_batch):
    imgs, gpss, inds, sids = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    inds_batched = torch.stack(list(inds), dim=0)
    sids_batched = torch.cat(sids, dim=0)
    return img_batched, gps_batched, inds_batched, sids_batched

def collate_multiscale(original_batch):
    (imgs_species, imgs_landscape, imgs_satellite, 
     gpss_species, gpss_landscape, gpss_satellite,
     inds, inds_species, inds_landscape, inds_satellite,
     ids_species, ids_landscape, ids_satellite,
     labels_species, labels_landscape, labels_satellite) = zip(*original_batch)

    inds = torch.stack(list(inds), dim=0)

    img_batched_species = torch.cat(list(imgs_species), dim=0)
    label_batches_species = torch.stack(list(labels_species), dim=0)
    gps_batched_species = torch.stack(list(gpss_species), dim=0)
    inds_batched_species = torch.stack(list(inds_species), dim=0)
    ids_batched_species = torch.stack(list(ids_species), dim=0)

    img_batched_landscape = torch.cat(list(imgs_landscape), dim=0)
    label_batches_landscape = torch.stack(list(labels_landscape), dim=0)
    gps_batched_landscape = torch.stack(list(gpss_landscape), dim=0)
    inds_batched_landscape = torch.stack(list(inds_landscape), dim=0)
    ids_batched_landscape = torch.stack(list(ids_landscape), dim=0)

    img_batched_satellite = torch.cat(list(imgs_satellite), dim=0)
    label_batched_satellite = torch.stack(list(labels_satellite), dim=0)
    gps_batched_satellite = torch.stack(list(gpss_satellite), dim=0)
    inds_batched_satellite = torch.stack(list(inds_satellite), dim=0)
    ids_batched_satellite = torch.stack(list(ids_satellite), dim=0)

    return {
        'indices': inds,
        'species': (img_batched_species, gps_batched_species, inds_batched_species, ids_batched_species, label_batches_species),
        'landscape': (img_batched_landscape, gps_batched_landscape, inds_batched_landscape, ids_batched_landscape, label_batches_landscape),
        'satellite': (img_batched_satellite, gps_batched_satellite, inds_batched_satellite, ids_batched_satellite, label_batched_satellite),
    }


class DataFrameMultiIdSampler(Sampler):
    """Custom sampler for iterating over non-unique IDs.
    
    Works on pandas DataFrames.
    """
    def __init__(
        self,
        df,
        colId_name="surveyId",
        random_state=42,
    ):
        self.colId_name = colId_name
        self.df = (
            df.sample(frac=1, random_state=random_state)
                .drop_duplicates(subset=[colId_name])
        )
        self.iterating_ids = list(self.df[colId_name].keys())

    def __iter__(self):
        for _ in range(self.num_samples):
            iterating_id = self.iterating_ids[_]
            yield iterating_id

    def __len__(self):
        return self.df.shape[0]

    
