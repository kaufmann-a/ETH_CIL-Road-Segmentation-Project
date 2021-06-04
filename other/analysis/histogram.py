import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from tqdm import tqdm

from other.analysis.analysis import read_images

SAVE_FOLDER = "histograms"
DEBUG = False


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


def plot_histogram_cdf(img_hist_per_channel, img_cfd_per_channel, title="", fname=None, dir="", bins=np.arange(256)):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist = img_hist_per_channel[c]
        img_cdf = img_cfd_per_channel[c]
        axes[c].plot(bins, img_hist / img_hist.max())
        axes[c].plot(bins, img_cdf)
        axes[c].set_ylabel(c_color)
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


if __name__ == '__main__':
    # test data
    dirEthTestImages = "../../data/test_images/"
    process_on_dataset(dirEthTestImages, "eth_data_test")

    # training data
    DIR = "../../data/training/"

    datasets_dir = ["eth_dataset", "jkfrie", "matejsladek", "alessiapacca", "osm_roadtracer", "ottawa"]

    for dataset in datasets_dir:
        print(dataset)
        directory = os.path.join(os.path.join(DIR, dataset), "original/images")
        process_on_dataset(directory, dataset)
