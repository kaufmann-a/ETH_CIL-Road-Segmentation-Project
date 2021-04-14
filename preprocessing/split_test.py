import math
import os

from PIL import Image, ImageChops

DIR_TEST = "../data/test_images/"


def get_crop_box(i, j, height, width, stride):
    """
     Determine crop box.

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


def get_cropped_images(image, stride=(400, 400)):
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


def patch_image_together(cropped_images, total_width=608, total_height=608, stride=(400, 400)):
    width = total_width
    height = total_height

    new_image = Image.new('RGB', (width, height))

    image_idx = 0
    for i in range(math.ceil(height / stride[0])):
        for j in range(math.ceil(width / stride[1])):
            left, upper, right, lower = get_crop_box(i, j, height, width, stride)

            new_image.paste(cropped_images[image_idx], (left, upper))
            # new_image.show()
            image_idx += 1

    return new_image


def load_test_images(imgDir='../data/test_images/', stride=(400, 400), sanity_check=False):
    """
    Loads images from a directory and creates multiple cropped images (that if patched together
     again equal the original image) according to the stride.

    :param imgDir:
    :param stride: size of cropped output images (width, height)
    :param sanity_check: check if cropped images form again the original image
    :return: list of cropped images
    """
    out_image_list = []

    for file in os.listdir(imgDir):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            input_image = Image.open(os.path.join(imgDir, filename))
            cropped_images = get_cropped_images(input_image, stride=stride)

            # concatenate out-images with new cropped-images
            out_image_list += cropped_images

            if sanity_check:
                img_patched = patch_image_together(cropped_images, stride=stride)
                if ImageChops.difference(input_image, img_patched).getbbox() is not None:
                    print("Images are not equal!")

    return out_image_list


if __name__ == '__main__':
    # test function
    images = load_test_images(DIR_TEST, stride=(400, 400), sanity_check=True)
    if not len(images) == 94 * 4:
        print("Image output count not as expected!")
