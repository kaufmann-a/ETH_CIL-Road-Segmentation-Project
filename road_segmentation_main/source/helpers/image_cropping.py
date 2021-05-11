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


class ImageCropper:
    def __init__(self, out_image_size=(400, 400), include_overlapping_patches=True):
        """

        :param out_image_size: the size of the extracted patches/images
        :param include_overlapping_patches: If the image size is not divisible by the out_image_size there
                                            will be (small) side areas of the image that do not build an entire patch.
                                            True = Include side area by shifting patch boundaries "into the image".
                                                   Notice: There are now overlapping areas between some of the patches.
                                            False = Discards side areas.
                                                    Notice: If for instance the image has size 600x600 and
                                                    out_image_size is 400x400 only one patch will be generated and
                                                    we loose an image area of 2*(200x400) + (200x200) pixels.
        """
        self.out_image_size = out_image_size
        self.include_overlapping_patches = include_overlapping_patches

    def get_cropped_image(self, image: PIL.Image, index_of_segment):
        """
        Get the "index_of_segment"-th crop of the supplied image.

        :param image:
        :param index_of_segment: should be zero indexed
        :return: cropped image
        """
        height = image.height
        width = image.width

        height_upper_bound, width_upper_bound = self.get_height_width_upper_bound_idx(height, width)
        if height_upper_bound * width_upper_bound <= index_of_segment:
            raise ValueError('Segment does not exist:', index_of_segment)

        i = index_of_segment // height_upper_bound
        j = index_of_segment % width_upper_bound

        box = get_crop_box(i, j, height, width, self.out_image_size)
        img_cropped = image.crop(box)

        return img_cropped

    def get_height_width_upper_bound_idx(self, height, width):
        height_upper_bound = height / self.out_image_size[0]
        width_upper_bound = width / self.out_image_size[1]

        if self.include_overlapping_patches:
            # take ceil to include overlapping patches
            height_upper_bound = math.ceil(height_upper_bound)
            width_upper_bound = math.ceil(width_upper_bound)
        else:
            # take floor
            height_upper_bound = math.floor(height_upper_bound)
            width_upper_bound = math.floor(width_upper_bound)

        return height_upper_bound, width_upper_bound

    def get_cropped_images(self, image: PIL.Image):
        """
        Splits image into multiple smaller cropped images of size according to stride.

        :param image: the PIL image
        :return: list of cropped images
        """
        height = image.height
        width = image.width

        cropped_images = []

        height_upper_bound, width_upper_bound = self.get_height_width_upper_bound_idx(height, width)

        for i in range(height_upper_bound):
            for j in range(width_upper_bound):
                box = get_crop_box(i, j, height, width, self.out_image_size)
                img_cropped = image.crop(box)

                # img_cropped.show()
                # from matplotlib import pyplot
                # pyplot.imshow(img_cropped)
                # pyplot.show()

                cropped_images.append(img_cropped)

        return cropped_images

    def get_number_of_cropped_images(self, image: PIL.Image):
        height_upper_bound, width_upper_bound = self.get_height_width_upper_bound_idx(image.height, image.width)
        return height_upper_bound * width_upper_bound
