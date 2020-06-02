import numpy as np
from scipy.io import wavfile
from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset


def load_wavefile(filepath):
    _, data = wavfile.read(filepath)
    return data


def load_train():
    return DatasetFolder(
        root="data/train/audio",
        loader=load_wavefile,
        extensions=".wav"
    )


def get_valid(trainset):
    with open("data/train/validation_list.txt") as valid_file:
        valid_list =  tuple(
            line.replace("\n", "").replace("/", "\\") 
            for line in valid_file.readlines())
    valid_indices = [idx for (idx, sample) in enumerate(trainset.samples) if sample[0].endswith(valid_list)]
    return Subset(trainset, valid_indices)


def split_random(trainset, frac):
    indices = np.arange(len(trainset))
    train_indices = np.random.choice(indices, size=int(len(indices) * frac))
    valid_indices = indices[~np.isin(indices, train_indices)]
    return Subset(trainset, train_indices), Subset(trainset, valid_indices)