from PIL import Image, ImageFilter

# Post processing config
TARGET_AREA_SIZE = 400
ROAD_ALPHA_RANGE = list(range(45, 55))
LABEL_ROAD = (255, 255, 255, 255)
LABEL_NOT_ROAD = (0, 0, 0, 255)


def isPixelRoad(pixel):
    # This could be refined more in the future, but should be robust enough for first tests
    return pixel[3] != 0


def getPixelLabel(pixel):
    return LABEL_ROAD if isPixelRoad(pixel) else LABEL_NOT_ROAD


def erodeImage(image, size, noIterations):
    for i in range(noIterations):
        image = image.filter(ImageFilter.MinFilter(size))
    return image


def dilateImage(image, size, noIterations):
    for i in range(noIterations):
        image = image.filter(ImageFilter.MaxFilter(size))
    return image


def hasRoad(image):
    return any(list(map(lambda pixel: pixel == LABEL_ROAD, image.getdata())))


# Currently, erosion and dilation are very specific due to the "bad"/coarse initial pixel labelling
def labelledRoadData(roadImage):
    # Make image binary
    newData = map(getPixelLabel, roadImage.getdata())
    roadImage.putdata(list(newData))

    # Erode image, gets rid of minor noise
    roadImage = erodeImage(roadImage, 5, 4)

    # Dilate image, makes roads wider and gets rid of road names inside the road
    roadImage = dilateImage(roadImage, 3, 13)

    return roadImage


def downSampleImage(image):
    return image.resize((TARGET_AREA_SIZE, TARGET_AREA_SIZE), Image.ANTIALIAS)
