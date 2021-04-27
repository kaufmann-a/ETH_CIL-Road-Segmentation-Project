#!/usr/bin/python
import math
import os

import numpy as np
from PIL import Image

# change file path to your submission file
label_file = '../road_segmentation_main/trainings/submission.csv'

# default parameters - no changed needed
h = 16
w = h
imgwidth = int(math.ceil((600.0 / w)) * w)
imgheight = int(math.ceil((600.0 / h)) * h)
nc = 3


def binary_to_uint8(img):
    """
    Convert an array of binary labels to a uint8.

    :param img:
    :return: converted img
    """
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def process_one_line(im, line, upscale=True):
    """
    Processes a single line.

    :param im:
    :param line:
    :return:
    """
    tokens = line.split(',')
    id = tokens[0]
    prediction = int(tokens[1])

    tokens = id.split('_')
    i = int(tokens[1])
    j = int(tokens[2])

    if upscale:
        je = min(j + w, imgwidth)
        ie = min(i + h, imgheight)

        if prediction == 0:
            adata = np.zeros((w, h))
        else:
            adata = np.ones((w, h))
    else:
        i //= h
        j //= w
        je = min(j + 1, imgwidth // 16)
        ie = min(i + 1, imgheight // 16)

        if prediction == 0:
            adata = np.zeros((1, 1))
        else:
            adata = np.ones((1, 1))

    im[j:je, i:ie] = binary_to_uint8(adata)

    return im


def reconstruct_from_labels(image_id):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        im = process_one_line(im, line)

    Image.fromarray(im).save('prediction_' + '%.3d' % image_id + '.png')

    return im


def reconstruct_all_from_file(submission_file, upscale=False):
    STR_NOT_IN_FILE = "str-not-in-file"

    f = open(submission_file)
    lines = f.readlines()

    if not upscale:
        out_img_w = imgwidth // 16
        out_img_h = imgheight // 16
    else:
        out_img_w = imgwidth
        out_img_h = imgheight

    image_id_str = STR_NOT_IN_FILE
    image_out_list = list()

    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            if not image_id_str == STR_NOT_IN_FILE:
                # append image to output list
                image_out_list.append({"id": image_id, "img": im})
            # new image begins -> update id
            image_id_str = line[:4]  # get first 4 symbols
            image_id = int(image_id_str[:3])
            # reset image
            im = np.zeros((out_img_w, out_img_h), dtype=np.uint8)

        im = process_one_line(im, line, upscale=upscale)

    # append also last image
    image_out_list.append({"id": image_id, "img": im})

    return image_out_list


if __name__ == '__main__':
    predicted_images = reconstruct_all_from_file(label_file)

    out_folder = "./submission"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for img in predicted_images:
        save_file_path = os.path.join(out_folder, 'pred_' + '%.3d' % img["id"] + '.png')
        Image.fromarray(img["img"]).save(save_file_path)
