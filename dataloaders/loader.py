import numpy as np
from scipy.io import wavfile
from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset
from librosa.feature import mfcc
from torchvision import transforms
from torch import Tensor
import os


class TestsetFolder(DatasetFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index): 
        original_tuple = super(TestsetFolder, self).__getitem__(index)
        # the image file path
        path = os.path.basename(self.samples[index][0])
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def load_wavefile(filepath):
    shape = (30, 45) # (embeding, length)
    rate, data = wavfile.read(filepath)
    if np.max(data) == 0:
        normalized = data.astype(np.float32)
    else:
        normalized = data.astype(np.float32) / np.max(data)
    result =  mfcc(normalized, sr=rate, n_mfcc=shape[0])
    if result.shape[1] < shape[1]:
        return np.transpose(np.concatenate([np.zeros((shape[0], shape[1] - result.shape[1])), result], axis=1))
    return np.transpose(result[:, :shape[1]])


def load_train():
    return DatasetFolder(
        root="data/train/audio",
        loader=load_wavefile,
        extensions=".wav",
        transform=transforms.Compose([
            Tensor
        ])
    )


def load_test():
    return TestsetFolder(
        root="data/test",
        loader=load_wavefile,
        extensions=".wav",
        transform=transforms.Compose([
            Tensor
        ])
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