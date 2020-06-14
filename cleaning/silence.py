from shutil import copyfile
import os
from scipy.io import wavfile
import random
import numpy as np


dst_dir = "./data/train/audio/silence"
os.makedirs(dst_dir, exist_ok=True)
noise_dir = "./data/train/_background_noise_"
noise_files = [
    wavfile.read(os.path.join(noise_dir, file))
    for file in os.listdir(noise_dir)
    if file.endswith(".wav")
]

random.seed(42)
for i in range(2000):
    tracks = random.choices(noise_files, k=2)

    track1_rate, track1_data = tracks[0]
    track2_rate, track2_data = tracks[1]

    if track1_rate == track2_rate:
        rate = track1_rate

        track1_idx = random.randint(0, len(track1_data) - rate)
        track1_sample = track1_data[track1_idx:track1_idx + rate]

        track2_idx = random.randint(0, len(track2_data) - rate)
        track2_sample = track2_data[track2_idx:track2_idx + rate]

        track1_c = random.random() % 1
        track2_c = 1 - track1_c
        track_mix = (track1_sample * track1_c + track2_sample * track2_c).astype(np.int16)

        filepath = os.path.join(dst_dir, "{}_noise_mix.wav".format(i))
        wavfile.write(filepath, rate, track_mix)
        print("{} created".format(filepath))