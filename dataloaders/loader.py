import numpy as np
from scipy.io import wavfile
from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset
from librosa.feature import mfcc


def load_wavefile(filepath):
    shape = (20, 40)
    rate, data = wavfile.read(filepath)
    normalized = data.astype(float) / np.max(data)
    result =  mfcc(normalized, sr=rate, n_mfcc=shape[0])
    if result.shape[1] < shape[1]:
        return np.concatenate([np.zeros((shape[0], shape[1] - result.shape[1])), result], axis=1)
    return result[:, :shape[1]]


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