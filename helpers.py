import torch
import torch.backends.cudnn
import numpy as np
import random


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
        Set seed for all torch func and methods.
        See ref: https://github.com/pytorch/pytorch/issues/7068
        """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
