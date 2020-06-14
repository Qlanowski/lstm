from shutil import copyfile
import os

dst_dir = "./data/train/audio/silence"
os.makedirs(dst_dir, exist_ok=True)

with open("./cleaning/bad_files.txt", "r") as f:
    files = f.read().split('\n')

cnt = 0
for file in files:
    src = os.path.join("./full_data/train/audio", file)
    file_name = "{}_{}".format(
        os.path.dirname(file),
        os.path.basename(file)
    )
    dst = os.path.join(dst_dir, file_name)
    if os.path.exists(src) and not os.path.exists(dst):
        copyfile(src, dst)
        cnt += 1
    else:
        print("NOT COPIED", src, dst)

print(f"copied {cnt}/{len(files)}")