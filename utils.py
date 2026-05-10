import random
import torch
import numpy as np
def set_seed(seed=42):
    """
    Set random seed for reproducibility across different libraries.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    random.seed(seed)  # Python built-in random number generator
    np.random.seed(seed)  # NumPy random number generator
    torch.manual_seed(seed)  # PyTorch CPU random number generator
    torch.cuda.manual_seed(seed)  # PyTorch GPU random number generator
    torch.cuda.manual_seed_all(seed)  # Ensure consistency for multi-GPU training
    torch.backends.cudnn.deterministic = True  # Ensure CUDA computation is reproducible
    torch.backends.cudnn.benchmark = False  # Disable acceleration for stable results