import albumentations as A
import cv2

#https://github.com/albumentations-team/albumentations#i-am-new-to-image-augmentation

# Declare an augmentation pipeline
transform_randomcrop = A.Compose([
    A.RandomCrop(width=400, height=400),
])

# Available interpolation flags: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR
# Available border_mode flags: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101
# BORDER_REPLICATE doesn't look so good.
# Rotate the input image by a random angle
transform_randomrotate = A.Compose([
    A.augmentations.geometric.rotate.Rotate(limit=360, interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, always_apply=False, p=1)
])

# upsample image to size 608X608
transform_upsample = A.Compose([
    A.augmentations.geometric.resize.LongestMaxSize(max_size=608, interpolation=cv2.INTER_NEAREST, always_apply=False, p=1),
])

# downsample image to size 400X400
transform_downsample = A.Compose([
    A.augmentations.geometric.resize.LongestMaxSize(max_size=400, interpolation=cv2.INTER_NEAREST, always_apply=False, p=1),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("input/satImage_004.png")

# Augment an image
transformed = transform_randomrotate(image=image)
transformed_image = transformed["image"]
cv2.imwrite("output/randomrotate3_satImage_004.png", transformed_image)

### Relevant image transformations.
# download relevant images. see which kind of transformations could be useful
# rotate image by a random angle
# maybe it is useful to do a linear interpolation of the mask?
# is it useful to have multiple masks as output and then give the output based on all of them and some kind of confidence interval.