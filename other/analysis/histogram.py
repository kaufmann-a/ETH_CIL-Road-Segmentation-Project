import os
import sys

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage import io
from skimage.exposure import match_histograms
from tqdm import tqdm

from other.analysis.analysis import read_images

SAVE_FOLDER = "histograms"
DEBUG = False


def plot_single_histogram(img, name):
    img_hist_per_c, img_cdf_per_c = get_histograms_single_image(img)
    plot_histogram_cdf(img_hist_per_c, img_cdf_per_c,
                       title=name,
                       fname=None,
                       dir=SAVE_FOLDER)


def get_histograms_single_image(img):
    img_hist_color_channels = list()
    img_hist_cdf_channels = list()
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist, bins = exposure.histogram(img[..., c], source_range='dtype', nbins=256)
        assert (np.all(np.arange(256) == bins))
        img_hist_color_channels.append(img_hist)

        # img_cdf, bins = exposure.cumulative_distribution(img[..., c], nbins=256)
        # try:
        #     assert (np.all(np.arange(256) == bins))
        # except:
        #     print(f"image nr {i}")

        img_cdf = img_hist.cumsum()
        img_cdf = img_cdf / float(img_cdf[-1])

        img_hist_cdf_channels.append(img_cdf)

    img_hist_per_c = np.vstack(img_hist_color_channels)
    img_cdf_per_c = np.vstack(img_hist_cdf_channels)

    return img_hist_per_c, img_cdf_per_c


def get_histograms(image_list, name="train", save_images=False, debug=False):
    img_hist_list = list()
    img_cdf_list = list()

    loop = tqdm(image_list, file=sys.stdout, postfix="get_histogram")

    for i, img in enumerate(loop):
        img_hist_per_c, img_cdf_per_c = get_histograms_single_image(img)

        img_hist_list.append(img_hist_per_c)
        img_cdf_list.append(img_cdf_per_c)

        if debug:
            plot_histogram_cdf(img_hist_per_c, img_cdf_per_c,
                               title=str(i),
                               fname=str(i) + "_hist" if save_images else None,
                               dir=os.path.join(SAVE_FOLDER, name))
            if debug == 2:
                plt.title(str(i))
                plt.imshow(img)
                plt.show()

    return img_hist_list, img_cdf_list


def plot_histogram_cdf(img_hist_per_channel, img_cfd_per_channel, title="", fname=None, dir="", bins=np.arange(256),
                       plot_cdf=False):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist = img_hist_per_channel[c]
        img_cdf = img_cfd_per_channel[c]
        axes[c].plot(bins, img_hist, color=c_color)
        if plot_cdf:
            axes[c].plot(bins, img_cdf)
        axes[c].set_ylabel("count")
        axes[c].set_xlabel(c_color + ": intensity value")
    fig.suptitle(title)
    if fname is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(os.path.join(dir, fname))
        plt.close(fig)
    else:
        plt.show()


def get_mean_hist(image_list, name="", save_mean_image=True, save_hist_images=False):
    hist, cdf = get_histograms(image_list, name=name, save_images=save_hist_images, debug=DEBUG)

    hist = np.asarray(hist)
    cdf = np.asarray(cdf)

    mean_hist = np.asarray(hist).mean(axis=0)
    mean_cdf = np.asarray(cdf).mean(axis=0)

    plot_histogram_cdf(mean_hist, mean_cdf,
                       title=name + " - mean histogram",
                       fname=name + "_mean_hist" if save_mean_image else None,
                       dir=SAVE_FOLDER)

    return mean_hist, mean_cdf


def process_on_dataset(dir, name, nr_images_to_process=None):
    img_list = read_images(dir, as_array=True)

    print("nr images", len(img_list))

    if nr_images_to_process is not None:
        img_list = img_list[:nr_images_to_process]

    return get_mean_hist(img_list, name=name)


def apply_augmentation(transform, image, text):
    image_trans = transform(image=image)["image"]
    plot_single_histogram(image_trans, text)
    plt.imshow(image_trans)
    plt.title(text)
    plt.show()


def test_histogram_matching():
    img_train = io.imread("../../data/training/eth_dataset/original/images/satImage_001.png")
    img_test = io.imread("../../data/test_images/test_7.png")

    plt.imshow(img_train)
    plt.title("train image")
    plt.show()

    plt.imshow(img_test)
    plt.title("test image")
    plt.show()

    # test matching
    plot_single_histogram(img_train, "train image")
    plot_single_histogram(img_test, "test image")

    matched = match_histograms(img_train, reference=img_test, multichannel=True)
    plot_single_histogram(matched, "matched image")
    plt.imshow(matched)
    plt.show()

    # apply color jitter
    t = A.ColorJitter(always_apply=True,
                      brightness=0.2,
                      contrast=0.2,
                      saturation=0.2,
                      hue=0.2)
    apply_augmentation(t, img_train, text="train image ColorJitter")
    apply_augmentation(t, img_test, text="test image ColorJitter")

    # apply random contrast
    t = A.RandomContrast(always_apply=True, limit=(0.2, 0.5))
    apply_augmentation(t, img_train, text="train image RandomContrast")
    apply_augmentation(t, img_test, text="test image RandomContrast")

    # apply gaussian noise to images
    t = A.GaussNoise(always_apply=True, var_limit=(50, 200))
    apply_augmentation(t, img_train, text="train image GaussNoise")
    apply_augmentation(t, img_test, text="test image GaussNoise")

    # equalize images
    t = A.Equalize(always_apply=True, by_channels=True)
    apply_augmentation(t, img_train, text="train image equalized")
    apply_augmentation(t, matched, text="matched image equalized")
    apply_augmentation(t, img_test, text="test image equalized")


if __name__ == '__main__':
    test_histogram_matching()
    # exit()

    # test data
    dirEthTestImages = "../../data/test_images/"
    process_on_dataset(dirEthTestImages, "ETH test images")

    # training data
    DIR = "../../data/training/"
    process_on_dataset(os.path.join(DIR, "eth_dataset/original/images"), "ETH train images")

    # other datasets
    datasets_dir = ["gmaps_public", "gmaps_custom", "matejsladek", "alessiapacca", "osm_roadtracer", "ottawa"]

    for dataset in datasets_dir:
        print(dataset)
        directory = os.path.join(os.path.join(DIR, dataset), "original/images")
        process_on_dataset(directory, dataset)
