import json
import os
import sys

from PIL import Image
from tqdm import tqdm

from src import gmaps_downloader
from src.osm_downloader_ import get_osm_roads
from src.tile_coordinates_helper import num2deg, deg2num

# bad masks from manual inspection
bad_masks = [
    'sat_62963_94290.png'
    'sat_62953_94291.png'
    'sat_62973_94322.png'
    'sat_79215_97557.png'
    'sat_79186_97535.png',
]


def get_sat_image(x, y, zoom, nr_tile_width, nr_tile_height, file):
    # get top left corner of top left tile (lat, lng)
    lat, lng = num2deg(x, y, zoom)

    # get top left corner of bottom right tile (lat, lng)
    lat_, lng_ = num2deg(x + nr_tile_width - 1, y + nr_tile_height - 1, zoom)

    gmaps_downloader.main(lng, lat, lng_, lat_, zoom=zoom, filePath=file, server="Google")


if __name__ == '__main__':
    data_folder_path = "./"
    recompute_masks = False
    location_files = ['locations_2.json', 'locations.json']

    data_folder_images = os.path.join(data_folder_path, "images")
    if not os.path.exists(data_folder_images):
        os.mkdir(data_folder_images)

    data_folder_masks = os.path.join(data_folder_path, "masks")
    if not os.path.exists(data_folder_masks):
        os.mkdir(data_folder_masks)

    # a tile has size 256x256 pixels
    nr_tiles_width = 7
    nr_tiles_height = 7

    zoom = 18

    for loc_file_name in location_files:
        with open(loc_file_name) as json_file:
            locations = json.load(json_file)

        loop = tqdm(locations, file=sys.stdout)
        for loc in loop:

            lat = loc['lat']  # 40.7161849
            lng = loc['lng']  # -111.8929752

            # get the top left tile coordinates (might be different to original lat, lng)
            x, y = deg2num(lat, lng, zoom)

            # file name for image and mask
            file_name = "sat_" + str(x) + "_" + str(y) + ".png"

            loop.set_postfix(file_name=file_name)

            # satellite image
            file_sat = os.path.join(data_folder_images, file_name)
            # satellite mask
            file_mask = os.path.join(data_folder_masks, file_name)

            if file_name in bad_masks:
                os.remove(file_sat)
                os.remove(file_mask)
                continue

            if not os.path.exists(file_sat):
                get_sat_image(x, y, zoom, nr_tiles_width, nr_tiles_height, file=file_sat)

            if not os.path.exists(file_mask) or recompute_masks:
                get_osm_roads(x, y, zoom, nr_tiles_width, nr_tiles_height, file_mask)

            # remove bad images/masks
            # TODO fix gmaps_downloader to give exact sized images
            img = Image.open(file_sat)
            if img.height != nr_tiles_width * 256 or img.width != nr_tiles_height * 256:
                print("Unexpected image width/height:", file_sat)
                os.remove(file_sat)
                os.remove(file_mask)
