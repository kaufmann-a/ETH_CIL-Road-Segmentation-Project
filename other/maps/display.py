import os
import sys

from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    dirImages = "./images"
    dirMasks = "./masks"

    loop = tqdm(os.listdir(dirImages), file=sys.stdout, postfix="read")
    for file in loop:
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            file_path = os.path.join(dirImages, filename)
            img = Image.open(file_path)

            img_size = 256 * 7
            if img.height != img_size or img.width != img_size:
                print("Unexpected image width/height:", file)

            file_path = os.path.join(dirMasks, filename)
            mask = Image.open(file_path)

            DPI = 100.0
            fig = plt.figure(figsize=(img.height / DPI, img.width / DPI))
            plt.title(file)
            plt.imshow(mask, alpha=0.4, zorder=1)
            plt.imshow(img, zorder=0)
            plt.show()
            plt.close(fig)
