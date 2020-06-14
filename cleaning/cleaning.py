import os

f = open("./cleaning/bad_files.txt", "r")
files = f.read().split('\n')

cnt = 0
for file in files:
    path = f"./data/train/audio/{file}"
    if os.path.exists(path):
        os.remove(path)
        cnt += 1

print(f"deleted {cnt}/{len(files)}")