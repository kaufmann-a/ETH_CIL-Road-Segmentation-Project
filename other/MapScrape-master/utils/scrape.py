import requests
import io
import math
import logging
import itertools
from PIL import Image
from utils.coordinateTransform import toTileCoordinates
from utils.MAPS_CONSTANTS import *


def tilesOffsetPerSide(length):
    noTilesNeeded = math.ceil((length / TILE_LENGTH))
    noTilesNeededNextUneven = noTilesNeeded + 1 if noTilesNeeded % 2 == 0 else noTilesNeeded
    return int((noTilesNeededNextUneven - 1) / 2)


def downloadTile(tileMode, tileX, tileY, tileZ):
    tileURL = MAPS_TILE_URL.format(tileMode, tileX, tileY, tileZ)
    tileData = requests.get(tileURL).content
    tileImage = Image.open(io.BytesIO(tileData))
    return tileImage


def downloadAreaSat(lat, lng, areaSize):
    centerTileX, centerTileY = toTileCoordinates(lat, lng)
    return downloadArea(centerTileX, centerTileY, areaSize, MAPS_MODE_SAT).convert('RGB')


def downloadAreaRoad(lat, lng, areaSize):
    centerTileX, centerTileY = toTileCoordinates(lat, lng)
    return downloadArea(centerTileX, centerTileY, areaSize, MAPS_MODE_ROAD)


def getTileCoordinatesForArea(lat, lng, areaSize):
    centerTileX, centerTileY = toTileCoordinates(lat, lng)
    offsetPerSide = tilesOffsetPerSide(areaSize)
    tileCoordinateOffsets = list(itertools.product(range(-offsetPerSide, offsetPerSide + 1), repeat=2))
    tileCoordinates = [(centerTileX + offsetX, centerTileY + offsetY) for (offsetX, offsetY) in tileCoordinateOffsets]
    return tileCoordinates


def downloadArea(centerTileX, centerTileY, areaSize, mode):
    # Initialize complete images
    areaImage = Image.new('RGBA', (areaSize, areaSize))

    # Calculate some stuff regarding tiles their positioning
    offsetPerSide = tilesOffsetPerSide(areaSize)
    lengthDownloaded = (2 * offsetPerSide + 1) * TILE_LENGTH
    overshotPerSide = int((lengthDownloaded - areaSize) / 2)

    # Calculate all tile coordinates needed for area
    tileCoordinateOffsets = itertools.product(range(-offsetPerSide, offsetPerSide + 1), repeat=2)

    # Download all needed tiles and stitch together
    for offsetX, offsetY in tileCoordinateOffsets:
        # Compute position of tile on final image
        tileBoxX = (offsetX + offsetPerSide) * TILE_LENGTH - overshotPerSide
        tileBoxY = (offsetY + offsetPerSide) * TILE_LENGTH - overshotPerSide
        tileBox = tileBoxX, tileBoxY

        # Download tile and paste on area image
        logging.debug('Downloading tiles')
        tileImage = downloadTile(mode, centerTileX + offsetX, centerTileY + offsetY, ZOOM_LEVEL)
        areaImage.paste(im=tileImage, box=tileBox)

    # Save images
    return areaImage
