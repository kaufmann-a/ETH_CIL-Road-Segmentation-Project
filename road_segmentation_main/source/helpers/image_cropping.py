"""
Helps splitting an image into multiple smaller images.
"""

import math

import PIL


def get_crop_box(i, j, height, width, stride):
    """
     Determine the crop box boundaries.

     :param i: x-axis index
     :param j: y-axis index
     :param height: total image height
     :param width: total image width
     :param stride: crop box size (x, y)
     :return: coordinates of top-left and bottom-right corner
    """
    # top left corner
    left = i * stride[0]  # x0
    upper = j * stride[1]  # y0
    # bottom right corner
    right = (i + 1) * stride[0]  # x1
    lower = (j + 1) * stride[1]  # y1

    if right > width:
        overlap = right - width
        right = width
        left -= overlap

    if lower > height:
        # shift box up
        overlap = lower - height
        lower = height
        upper -= overlap

    return left, upper, right, lower


def get_cropped_images(image: PIL.Image, stride=(400, 400)):
    """
    Splits image into multiple smaller cropped images of size according to stride.

    :param image: the PIL image
    :param stride: the output image size
    :return: list of cropped images
    """
    height = image.height
    width = image.width

    cropped_images = []

    for i in range(math.ceil(height / stride[0])):
        for j in range(math.ceil(width / stride[1])):
            box = get_crop_box(i, j, height, width, stride)
            img_cropped = image.crop(box)

            # img_cropped.show()
            # from matplotlib import pyplot
            # pyplot.imshow(img_cropped)
            # pyplot.show()

            cropped_images.append(img_cropped)

    return cropped_images
