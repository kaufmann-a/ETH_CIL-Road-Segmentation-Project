"""
Visualization of the validation set
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re
import torchmetrics as torchmetrics
import torch
import argparse

from source.helpers import filehelper

def process_args(args):
    if args.run_folder:
        args.workingdir = filehelper.build_abspath(args.run_folder, os.getcwd()) #In case relative path was passed in args

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Does prediction on predefined set of images"
    )
    parser.add_argument('--run_folder', default='./trainings/20210328-185846-default',
                        type=str, help="Input here the folder path of the training run you want to use for inference")

    args = parser.parse_args()
    process_args(args)

    valDir = os.path.join(args.workingdir, 'prediction-validation-set')
    accuracy = torchmetrics.Accuracy(threshold=0.5)

    for filename in os.listdir(valDir):
        if filename.endswith(".png") and len(re.findall(r"true_*", filename)) > 0:
            idx = filename.split(sep="_")[-1]
            filename_pred = "pred_" + idx
            filename_input = "input_" + idx

            true_img = Image.open(os.path.join(valDir, filename))
            pred_img = Image.open(os.path.join(valDir, filename_pred))
            input = Image.open(os.path.join(valDir, filename_input))

            true = np.where(np.array(true_img) > 255 * 0.5, 1, 0)
            pred = np.where(np.array(pred_img) > 255 * 0.5, 1, 0)

            true = torch.tensor(true)
            pred = torch.tensor(pred)

            acc = accuracy(true, pred)

            f, ax = plt.subplots(1,3)

            ax[0].imshow(true_img)
            ax[1].imshow(pred_img)
            ax[2].imshow(input)
            plt.tight_layout() # optional
            title = "Accuracy: " + str(np.array(acc))
            f.suptitle(title)
            plt.savefig(os.path.join(valDir, "summary_" + idx))
            plt.close()