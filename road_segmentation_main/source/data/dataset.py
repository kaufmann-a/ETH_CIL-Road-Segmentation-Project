__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from source.helpers.image_cropping import ImageCropper
from source.helpers.maskconverthelper import mask_to_submission_mask
from source.logcreator.logcreator import Logcreator


class RoadSegmentationDataset(Dataset):
    def __init__(self, engine, image_list, mask_list, threshold, transform=None,
                 crop_size=(400, 400),
                 use_submission_masks=False,
                 min_road_percentage=0.0001,
                 include_overlapping_patches=True,
                 debug=False):
        # self.device = device # unsure whether we need this, if yes: add parameter device to init
        self.transform = transform
        self.images = image_list
        self.images_filtered = []
        self.masks = mask_list
        self.foreground_threshold = threshold
        self.crop_size = crop_size
        self.use_submission_masks = use_submission_masks
        self.min_road_percentage = min_road_percentage if 1 >= min_road_percentage >= 0 else 0
        self.image_cropper = ImageCropper(out_image_size=crop_size,
                                          include_overlapping_patches=include_overlapping_patches)

        # preload images into memory to not read from drive everytime
        self.images_preloaded = list()
        self.masks_preloaded = list()

        Logcreator.info("Removing images that contain less than,",
                        crop_size[0] * crop_size[1] * self.min_road_percentage,
                        f"road pixels (less than {self.min_road_percentage}% of the image is road)")

        count_removed_images = 0

        loop = tqdm(zip(self.images, self.masks), total=len(self.images), file=sys.stdout, desc="Preload images")
        for image_path, mask_path in loop:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # get cropped images and masks
            images_cropped = self.image_cropper.get_cropped_images(image)
            masks_cropped = self.image_cropper.get_cropped_images(mask)

            images = [np.array(_img) for _img in images_cropped]
            masks = [np.array(_mask) for _mask in masks_cropped]

            if hasattr(engine, 'lines_layer_path'):
                lines_layer_cropped = self.image_cropper.get_cropped_images(Image.open(os.path.join(engine.lines_layer_path, os.path.basename(image_path))).convert('L'))
                predicted_layer_cropped = self.image_cropper.get_cropped_images(Image.open(os.path.join(engine.predicted_masks_path, os.path.basename(image_path))).convert('L'))
                lines_layer_imgs = [np.array(_img) for _img in lines_layer_cropped]
                predicted_layer_imgs = [np.array(_img) for _img in predicted_layer_cropped]

                for img_idx, img in enumerate(images):
                    lines_image_resh = np.reshape(lines_layer_imgs[img_idx], (400, 400, 1))
                    predicted_mask_image_resh = np.reshape(predicted_layer_imgs[img_idx], (400, 400, 1))
                    images[img_idx] = np.append(images[img_idx], lines_image_resh, axis=2)
                    images[img_idx] = np.append(images[img_idx], predicted_mask_image_resh, axis=2)

            if self.min_road_percentage > 0:
                # remove images with less percentage of road the given threshold
                keep_idx = [(np.sum(_mask > 0) / np.size(_mask) >= self.min_road_percentage) for _mask in masks]
                count_removed_images += np.sum(np.asarray(keep_idx) == False)

                if debug:
                    for _img, _mask, keep in zip(images, masks, keep_idx):
                        if keep == False:
                            plt.imshow(_img)
                            plt.show()
                            plt.imshow(_mask)
                            plt.show()

                images = [_img for _img, keep in zip(images, keep_idx) if keep]
                masks = [_mask for _mask, keep in zip(masks, keep_idx) if keep]
                if any(keep_idx):
                    self.images_filtered.append(image_path)
            else:
                self.images_filtered = self.images

            # concatenate lists
            self.images_preloaded += images
            self.masks_preloaded += masks

            # set to one since we preload all sub-patches
            self.nr_segments_per_image = 1

        Logcreator.info("Nr. of cropped image in total:", str(len(self.images_preloaded)))
        Logcreator.info("Nr. of removed images:", str(count_removed_images))

    def __len__(self):

        return len(self.images_preloaded)

    def __getitem__(self, index):
        image = self.images_preloaded[index]
        mask = self.masks_preloaded[index]

        threshold = 255.0 * self.foreground_threshold
        mask = (mask >= threshold).astype(int)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

            # 2d to 3d tensor
            mask = mask.unsqueeze(0).float()

            if self.use_submission_masks:  # use submission masks for training
                mask = mask_to_submission_mask(mask, threshold=self.foreground_threshold)

        return image, mask


class RoadSegmentationDatasetInference(Dataset):
    def __init__(self, engine, image_list, transform, crop_size=(400, 400), sanity_check=False):
        self.image_paths = image_list
        self.transform = transform
        self.crop_size = crop_size
        self.preloaded_images = []
        self.image_number_list = []

        image_cropper = ImageCropper(out_image_size=crop_size)

        # preload test images
        for file in image_list:
            filename = os.path.basename(file)
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # get image number
                self.image_number_list.append([int(s) for s in filename[:-4].split("_") if s.isdigit()][0])

                # get cropped images
                input_image = Image.open(file)
                cropped_images = image_cropper.get_cropped_images(input_image)

                if hasattr(engine, 'lines_layer_path') and engine.lines_layer_path is not None:
                    # load prediction + lines, append to image
                    preds_dir = engine.predicted_masks_path
                    lines_dir = engine.lines_layer_path
                    pred_filename = "pred_" + str(self.image_number_list[-1]) + ".png"

                    lines_image = Image.open(os.path.join(lines_dir, pred_filename)).convert('L')
                    pred_image = Image.open(os.path.join(preds_dir, pred_filename)).convert('L')

                    lines_layer_cropped = image_cropper.get_cropped_images(lines_image)
                    predicted_layer_cropped = image_cropper.get_cropped_images(pred_image)

                    lines_layer_imgs = [np.array(_img) for _img in lines_layer_cropped]
                    predicted_layer_imgs = [np.array(_img) for _img in predicted_layer_cropped]

                    for img_idx, img in enumerate(cropped_images):
                        # explicitly add new dimension (608,608) -> (608,608,1)
                        lines_image_resh = lines_layer_imgs[img_idx][:, :, np.newaxis]
                        predicted_mask_image_resh = predicted_layer_imgs[img_idx][:, :, np.newaxis]
                        cropped_images[img_idx] = np.append(cropped_images[img_idx], lines_image_resh, axis=2)
                        cropped_images[img_idx] = np.append(cropped_images[img_idx], predicted_mask_image_resh, axis=2)

                # concatenate out-images with new cropped-images
                self.preloaded_images += cropped_images

                if sanity_check:
                    img_patched = ImageCropper.patch_image_together(cropped_images, mode='RGB', stride=crop_size)
                    if ImageChops.difference(input_image, img_patched).getbbox() is not None:
                        print("Images are not equal!")

        self.nr_crops_per_image = int(len(self.preloaded_images) / len(self.image_paths))

    def __len__(self):
        return len(self.preloaded_images)

    def __getitem__(self, index):
        augmentations = self.transform(image=np.array(self.preloaded_images[index]))

        return augmentations["image"]


class SimpleToTensorDataset(Dataset):
    def __init__(self, image_path_list):
        self.img_path_list = image_path_list
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = Image.open(self.img_path_list[index]).convert("RGB")
        img = np.array(img)
        return self.transform(img)


class EnsembleDataset(Dataset):
    def __init__(self, image_lists):
        self.image_lists = image_lists # list of lists with images
        self.num_predictions = len(image_lists)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.shape = np.array(image_lists[0][0]).shape

    def __len__(self):
        return len(self.image_lists[0])

    def __getitem__(self, index):
        avg_image = np.zeros(self.shape)

        for image_list in self.image_lists:
            image = np.array(image_list[index]).astype(int)
            avg_image += image
        avg_image = avg_image / self.num_predictions
        avg_image = avg_image / 255 # transform to [0,1]

        return self.transform(avg_image)
