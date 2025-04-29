"""Main script to run training or inference on JRC multicale datasets.

Author: Theo Larcher <theo.larcher@inria.fr>
"""
from types import SimpleNamespace
from typing import Any, List

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import CenterCrop, Resize

from malpolon.data.datasets.jrc_multiscale import (
    LandscapeDatasetSimple,
    SatelliteDatasetSimple,
    SpeciesDatasetSimple,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_contrastive_system import (
    SimCLR,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_model import (
    ModelSimCLR,
)


# To address inconsistent image sizes, two options:
# 1. Define transforms to resize images to a fixed size
def transforms_species():
    def CenterCropToMaxDim(img):
        max_dim = max(img.shape[-2:])
        return CenterCrop((max_dim, max_dim))(img)

    ts = [lambda x: CenterCropToMaxDim(x),
          Resize((518, 518))]  # bilinear by default

    return transforms.Compose(ts)

# 2. Custom collate function returning directly a list of dictionaries with {'img': img_tensor, 'gps': gps_tuple}. But this implies adding a loop over the multi-dimensional tensors which defeats the purpose of batching.
def custom_collate(original_batch):
    imgs, gpss = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    return img_batched, gps_batched

def main(args):
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # Hyperparameters
    learning_rate = 0.00025
    num_epochs = 10

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # dataset = ContrastiveLearningDataset(args.data)
    dataset = SpeciesDatasetSimple(
        root_path = 'dataset/scale_1_species/data_subset/img',
        fp_metadata = 'dataset/scale_1_species/data_subset/metadata_subset.csv',
        transform = transforms_species(),
    )  # dataset_species, dataset_landscape, dataset_satellite

    # train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    train_dataset = dataset

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)

    model = ModelSimCLR(base_model=args.arch, out_dim=args.out_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=25)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    args_ns = {
        'arch': 'species',  # always paired with gps
        'epochs': 10,
        'out_dim': 512,
        'batch_size': 5,
        'n_views': 2,  # must be equal to the number of modalities passed to the contrastive loss
        'temperature': 0.1,
        'fp16_precision': False,
        'disable_cuda': False,
        'log_every_n_steps': 10,
        'workers': 0,
        'gpu_index': 0,
        'disable_cuda': True,
    }
    args_ns = SimpleNamespace(**args_ns)
    main(args_ns)
