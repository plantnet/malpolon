"""This module checks the installation of PyTorch and GPU libraries.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
"""

import os

import torch


def print_cuda_info():
    """Print information about the CUDA/PyTorch installation."""
    print(f"Using PyTorch version {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()} (version: {torch.version.cuda})")
    print(f"cuDNN available: {torch.backends.cudnn.enabled} (version: {torch.backends.cudnn.version()})")
    print(f"Number of CUDA-compatible devices found: {torch.cuda.device_count()}")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")


if __name__ == "__main__":
    print_cuda_info()
