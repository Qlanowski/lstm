import os

from shutil import copyfile, rmtree
import os

dst_dir = './data/train/audio/ALL/'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

valid ={"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
        "silence", "ALL"}
cnt = 0
for x in os.listdir('./data/train/audio/'):
    print(x)
    if not (x in valid):
        for file in os.listdir(f'./data/train/audio/{x}'):
            src = f"./data/train/audio/{x}/{file}"
            file_name = file.split("/")[-1]
            dst = f"./data/train/audio/ALL/{x}_{file}"
            copyfile(src, dst)
            cnt += 1
        rmtree(f'./data/train/audio/{x}')