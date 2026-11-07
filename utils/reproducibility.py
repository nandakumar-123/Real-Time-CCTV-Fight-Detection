"""
Reproducibility and Determinism Configuration.
Ensures identical seed initialization across random, numpy, and PyTorch.
"""

import os
import random
import numpy as np

def seed_everything(seed: int = 42) -> int:
    """
    Sets random seeds for reproducibility across Python random, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed

