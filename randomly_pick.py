import os
import random
import shutil

random.seed(32)

root_folder = "dataset"
folder_path = "dataset/images"

# List all files (or filter by extension)
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Shuffle and pick 20%
num_to_select = int(len(all_files) * 0.1)
random_10_files = random.sample(all_files, num_to_select)
remaining_90_files = [f for f in all_files if f not in random_10_files]

for file in random_10_files:
    shutil.copy(os.path.join(folder_path, file), os.path.join(folder_path, "val", file))
    shutil.copy(os.path.join(root_folder, 'labels', file.replace('.png', '.txt')), os.path.join(root_folder, 'labels', "val", file.replace('.png', '.txt')))
for file in remaining_90_files:
    shutil.copy(os.path.join(folder_path, file), os.path.join(folder_path, "train", file))
    shutil.copy(os.path.join(root_folder, 'labels', file.replace('.png', '.txt')), os.path.join(root_folder, 'labels', "train", file.replace('.png', '.txt')))