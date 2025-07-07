"""
Memory monitoring utility functions for the medical image analysis project.
"""

import torch
import psutil
import gc


def print_memory_stats():
    """
    Print current system and GPU memory statistics.
    """
    # System memory
    memory = psutil.virtual_memory()
    print(f"System Memory - Available: {memory.available / (1024**3):.2f} GB, Used: {memory.percent:.1f}%")
    
    # GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("CUDA not available. No GPU memory stats to show.")


def clear_memory():
    """
    Clear memory caches and run garbage collection.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 