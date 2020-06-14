from shutil import copyfile
import os

dst_dir = './data/train/audio/silence/'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

f = open("./cleaning/bad_files.txt", "r")
files = f.read().split('\n')

cnt = 0
for file in files:
    src = f"./full_data/train/audio/{file}"
    file_name = file.split("/")[-1]
    dst = f"{dst_dir}{file_name}"
    if os.path.exists(src) and not os.path.exists(dst):
        copyfile(src, dst)
        cnt += 1

print(f"copied {cnt}/{len(files)}")