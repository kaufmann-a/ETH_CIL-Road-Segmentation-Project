import math
from math import pi as PI
from utils.MAPS_CONSTANTS import TILE_LENGTH, ZOOM_LEVEL


def toTileCoordinates(lat, lng):
    xProjected, yProjected = mercatorProject(lat, lng)
    scale = 2**ZOOM_LEVEL
    xTile = math.floor(xProjected * scale / TILE_LENGTH)
    yTile = math.floor(yProjected * scale / TILE_LENGTH)
    return xTile, yTile


def mercatorProject(lat, lng):
    siny = math.sin(lat * PI / 180)
    xCoord = TILE_LENGTH * (0.5 + lng / 360)
    yCoord = TILE_LENGTH * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * PI))
    return xCoord, yCoord


if __name__ == '__main__':
    print(toTileCoordinates(41.85, -87.65))