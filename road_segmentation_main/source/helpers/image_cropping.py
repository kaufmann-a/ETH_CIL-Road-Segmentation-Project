"""
Helps splitting an image into multiple smaller images.
"""

import math

import PIL
import torch
from PIL import Image
from matplotlib import pyplot


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

    @staticmethod
    def patch_image_together(cropped_images, mode='RGB', total_width=608, total_height=608, stride=(400, 400)):
        width = total_width
        height = total_height

        new_image = Image.new(mode, (width, height))

        image_idx = 0
        for i in range(math.ceil(height / stride[0])):
            for j in range(math.ceil(width / stride[1])):
                left, upper, right, lower = get_crop_box(i, j, height, width, stride)

                new_image.paste(cropped_images[image_idx], (left, upper))
                # new_image.show()
                image_idx += 1

        return new_image

    @staticmethod
    def patch_masks_together(cropped_images, device, out_image_size=(608, 608), stride=(400, 400),
                             mode='avg', debug=False):
        """
        Stitches a image mask together from multiple cropped image masks.

        :param cropped_images: list of 2d-image tensors
        :param device: cuda or cpu
        :param out_image_size: (output image width, output image height)
        :param stride: size of cropped images
        :param mode: 'avg': take the average of the overlapping areas,
                     'max': take the maximum of the overlapping areas,
                     'overwrite': overwrite the overlapping areas
        :param debug: True = plot images

        :return: the combined image
        """
        width = out_image_size[0]
        height = out_image_size[1]

        out_array = torch.zeros(size=(width, height)).to(device)
        out_overlap_count = torch.ones(size=(width, height)).to(device)  # stores nr of intersection at pixels
        mem_right = mem_lower = right = lower = 0

        image_idx = 0
        for i in range(math.ceil(height / stride[0])):
            for j in range(math.ceil(width / stride[1])):
                mem_lower = lower if mem_lower < lower < height else mem_lower
                mem_right = right if mem_right < right < width else mem_right
                left, upper, right, lower = get_crop_box(i, j, height, width, stride)

                if mode == 'avg':
                    # get overlaping ranges
                    overlapping_indices_2 = torch.nonzero(out_array[upper:lower, left:right], as_tuple=True)
                    # Check if there is any overlap at all
                    if len(overlapping_indices_2[0]) > 0 and len(overlapping_indices_2[1]) > 0:
                        upper_overlap = int(torch.min(overlapping_indices_2[0])) + upper
                        lower_overlap = int(torch.max(overlapping_indices_2[0])) + upper + 1
                        left_overlap = int(torch.min(overlapping_indices_2[1])) + left
                        right_overlap = int(torch.max(overlapping_indices_2[1])) + left + 1

                        out_overlap_count[upper_overlap:lower_overlap, left_overlap:right_overlap] += 1

                        # case bottom right image was placed, we need to subtract bottom right square again
                        if torch.max(out_overlap_count) == 4:
                            out_overlap_count[mem_lower:lower, mem_right:right] -= 1

                    out_array[upper:lower, left:right] = torch.add(out_array[upper:lower, left:right],
                                                                   cropped_images[image_idx])
                elif mode == 'max':
                    out_array[upper:lower, left:right] = torch.maximum(out_array[upper:lower, left:right],
                                                                       cropped_images[image_idx])
                else:
                    # overwrite overlapping areas
                    out_array[upper:lower, left:right] = cropped_images[image_idx]

                if debug:
                    # print(left, upper, right, lower)
                    # print(cropped_images[image_idx].shape)
                    plot_array = out_array > 0.5
                    pyplot.imshow(plot_array.cpu(), cmap='gray', vmin=0, vmax=1)
                    pyplot.show()

                image_idx += 1
        if mode == 'avg':
            out_array = torch.div(out_array, out_overlap_count)
        return out_array
