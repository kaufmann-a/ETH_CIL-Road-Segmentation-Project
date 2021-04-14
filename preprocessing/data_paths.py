import os

DIR = "../data/training"

folders = ["original", "flip_hor", "flip_ver", "crop_random", "rotate_random"]

# train test split here (original only or all), and then write only the remaining paths to file

image_paths = []
mask_paths = []

for folder in folders:
    for file in os.listdir(os.path.join(DIR, folder, "images")):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            image_paths.append(os.path.join(folder, "images", filename))
            mask_paths.append(os.path.join(folder, "masks", filename))

with open("image_paths.txt", "w") as file:
    for row in image_paths:
        file.write(row + '\n')

with open("mask_paths.txt", "w") as file:
    for row in mask_paths:
        file.write(row + '\n')