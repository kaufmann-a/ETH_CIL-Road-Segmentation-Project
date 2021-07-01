import json
import os
import logging
import time
import datetime
from PIL import Image
from utils.scrape import downloadAreaSat, downloadAreaRoad, getTileCoordinatesForArea
from utils.postprocess import labelledRoadData, downSampleImage, hasRoad

# FS config
DIR_PATH_SAT = './mined/images/'
DIR_PATH_ROAD = './mined/masks/'
FILE_NAME_SAT = 'extra_sat_{}_{}.png'
FILE_NAME_ROAD = 'extra_sat_{}_{}.png'
COORD_SIGNIFICANT_DECIMAL = 7

# Scraping config
FILE_PATH_LOCATIONS = '../maps/locations_2.json'
AREA_SIZE = 1250
NEIGHBOR_COUNT = 3
NEIGHBOR_COORD_OFFSET = 0.0035


def configureLogging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')


def anyTileAlreadyDownloaded(downloadedTiles, tileToDownload):
    for tile in tileToDownload:
        if tile in downloadedTiles:
            return True

    return False


def readLocationsFromFile(locationsFilePath):
    with open(locationsFilePath) as f:
        return json.load(f)


def formattedCoord(coord):
    return round(coord, COORD_SIGNIFICANT_DECIMAL)


def downloadAndProcessSat(lat, lng, filePathSat):
    if os.path.isfile(filePathSat):
        logging.warning('Sat data is already downloaded. Skipping.')
        return Image.open(filePathSat)

    areaImageSat = downloadAreaSat(lat, lng, AREA_SIZE)
    areaImageSat = downSampleImage(areaImageSat)
    areaImageSat.save(filePathSat)
    return areaImageSat


def downloadAndProcessRoad(lat, lng, filePathRoad):
    if os.path.isfile(filePathRoad):
        logging.warning('Road data is already downloaded. Skipping.')
        return Image.open(filePathRoad)

    areaImageRoad = downloadAreaRoad(lat, lng, AREA_SIZE)
    areaImageRoad = labelledRoadData(areaImageRoad)
    areaImageRoad = downSampleImage(areaImageRoad)
    areaImageRoad.save(filePathRoad)
    return areaImageRoad


def main():
    # Save time and set up logger
    timeStart = time.time()
    configureLogging()

    # Remembers downloaded tiles
    downloadedTiles = set()
    areasSkippedBecauseOfDuplicatedTiles = 0
    areasSkippedBecauseOfNoRoad = 0

    # Create directories for scraped data if it doesn't exist yet
    logging.info('Creating target directories if they don\'t exist yet')
    os.makedirs(DIR_PATH_SAT, exist_ok=True)
    os.makedirs(DIR_PATH_ROAD, exist_ok=True)

    # Read locations from json file
    logging.info('Reading locations')
    targetLocations = readLocationsFromFile(FILE_PATH_LOCATIONS)
    noTargetLocations = len(targetLocations)
    logging.info('There are {} locations specified'.format(noTargetLocations))

    if NEIGHBOR_COUNT > 0:
        noLocationsIncludingNeighboringAreas = noTargetLocations * (NEIGHBOR_COUNT * 2 + 1)**2
        logging.info('With neighboring areas, there are {} locations.'.format(noLocationsIncludingNeighboringAreas))

    logging.info('Starting downloads')
    for index, targetLocation in enumerate(targetLocations):
        targetLat, targetLng = targetLocation['lat'], targetLocation['lng']

        for latOffset in range(-NEIGHBOR_COUNT, NEIGHBOR_COUNT + 1):
            for lngOffset in range(-NEIGHBOR_COUNT, NEIGHBOR_COUNT + 1):
                # Compute new lat and lng of specific neighbor area
                lat = formattedCoord(targetLat + latOffset * NEIGHBOR_COORD_OFFSET)
                lng = formattedCoord(targetLng + lngOffset * NEIGHBOR_COORD_OFFSET)

                # Check if any tile in this area has already been downloaded to avoid duplicated data
                tilesToDownload = getTileCoordinatesForArea(lat, lng, AREA_SIZE)
                if anyTileAlreadyDownloaded(downloadedTiles, tilesToDownload):
                    logging.warning('Data at {} / {} contains tiles which are already used. Skipping.'.format(lat, lng))
                    areasSkippedBecauseOfDuplicatedTiles += 1
                    continue

                # Remember downloaded tiles
                for tile in tilesToDownload:
                    downloadedTiles.add(tile)

                # Download satellite and road data
                progress = '{} / {}'.format(index, noTargetLocations)
                logging.info('{}: Downloading data at {} / {}'.format(progress, lat, lng))
                filePathSat = DIR_PATH_SAT + FILE_NAME_SAT.format(lat, lng)
                filePathRoad = DIR_PATH_ROAD + FILE_NAME_ROAD.format(lat, lng)

                downloadAndProcessSat(lat, lng, filePathSat)
                roadImage = downloadAndProcessRoad(lat, lng, filePathRoad)

                # Check if image even has road, otherwise delete
                if not hasRoad(roadImage):
                    logging.warning('Data does not contain any road. Deleting files again...')
                    areasSkippedBecauseOfNoRoad += 1
                    os.remove(filePathSat)
                    os.remove(filePathRoad)

    # Some additional information at the end
    timeEnd = time.time()
    timeElapsed = datetime.timedelta(seconds=timeEnd - timeStart)
    logging.info('All files downloaded, time elapsed: {}'.format(str(timeElapsed)))
    logging.info('Skipped {} areas because of duplicated tiles.'.format(areasSkippedBecauseOfDuplicatedTiles))
    logging.info('Skipped {} areas because they did not contain roads.'.format(areasSkippedBecauseOfNoRoad))


if __name__ == '__main__':
    main()
