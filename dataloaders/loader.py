from scipy.io import wavfile
from torchvision.datasets import DatasetFolder


def load_wavefile(filepath):
    _, data = wavfile.read(filepath)
    return data


def load_train():
    return DatasetFolder(
        root="data/train/audio",
        loader=load_wavefile,
        extensions=".wav"
    )