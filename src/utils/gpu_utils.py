"""
GPU-related utility functions for the medical image analysis project.
"""

import torch
import warnings
import logging


def configure_gpu():
    """
    Configure GPU settings and display information.
    
    Returns:
        torch.device: The device to use (cuda or cpu)
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    return device


def print_gpu_memory_stats():
    """
    Print current GPU memory statistics.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    else:
        print("CUDA not available. No GPU memory stats to show.") 