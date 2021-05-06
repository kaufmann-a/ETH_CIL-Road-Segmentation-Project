import numpy as np
import cv2

# this is function for continuous dilations followed by erosions
def erodedilateallones(img, numdilate=4, numerode=4, nummedian=2, kernelsz=16, mediansz=5 ):
    kernel_allones = np.ones((kernelsz,kernelsz))
    #Continuous dilations. Spreads out the pixels to join the roads.
    kernel_allones = kernel_allones.astype('uint8')
    if(numdilate>0):
        out_img = cv2.dilate(img, kernel_allones)
        for i in range(0,numdilate-1):
            out_img = cv2.dilate(out_img, kernel_allones)
    #Continuous erosion. Clears up unwanted pixels which are not part of road
    # Ideally numerode should be equal to numdilate to maintain the pixel resolution
    if(numerode>0):
        out_img = cv2.erode(out_img, kernel_allones)
        for i in range(0,numerode-1):
            out_img = cv2.dilate(out_img, kernel_allones)
    #Median filtering to sharpen the edges
    if(nummedian>0):
        out_img = cv2.medianBlur(out_img, mediansz)
        for i in range(0,numerode-1):
            out_img = cv2.medianBlur(out_img, mediansz)

    return out_img