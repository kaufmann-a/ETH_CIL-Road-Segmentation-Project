import numpy as np
import cv2

# this is function for continuous dilations followed by erosions
# to maintain road width pixel resolution set numdilate = numerode
def erodedilate_v0(img, numdilate=4, numerode=4, nummedian=2, kernelsz=16, mediansz=5 ):
    kernel_allones = np.ones((kernelsz,kernelsz))
    #Continuous dilations. Spreads out the pixels to join the roads.
    kernel_allones = kernel_allones.astype('uint8')
    if(numdilate>0):
        out_img = cv2.dilate(img, kernel_allones)
        for i in range(0,numdilate-1):
            out_img = cv2.dilate(out_img, kernel_allones)
    #Continuous erosion. Clears up unwanted pixels which are not part of road
    if(numerode>0):
        out_img = cv2.erode(out_img, kernel_allones)
        for i in range(0,numerode-1):
            out_img = cv2.erode(out_img, kernel_allones)

    #Median filtering to sharpen the edges
    if(nummedian>0):
        out_img = cv2.medianBlur(out_img, mediansz)
        for i in range(0,numerode-1):
            out_img = cv2.medianBlur(out_img, mediansz)

    return out_img


# this is function for continuous dilations followed by erosions
# to maintain road width pixel resolution set numdilate = numerode
def erodedilate_v1(img, numdilate=4, numerode=4, nummedian=2, kernelsz=16, mediansz=5 ):
    kernel_1 = np.zeros((kernelsz,kernelsz))

    # central element 1s
    kernel_010 = np.zeros((kernelsz,kernelsz))
    midpt = int(kernelsz/2)
    kernel_010[:,[midpt]]  = 1

    # rows central element 1s
    kernel_010_r = np.zeros((kernelsz,kernelsz))
    kernel_010_r[[midpt],:] = 1

    # diagonal elements 1s
    kernel_diag = np.zeros((kernelsz,kernelsz))
    np.fill_diagonal(kernel_diag,1)

    # flipped diagonal 1s
    kernel_diag_flip = np.fliplr(kernel_diag)

    #Continuous dilations. Spreads out the pixels to join the roads.
    kernel_010 = kernel_010.astype('uint8')
    kernel_010_r = kernel_010_r.astype('uint8')
    kernel_diag = kernel_diag.astype('uint8')
    kernel_diag_flip = kernel_diag_flip.astype('uint8')

    if(numdilate>0):
        out_img = cv2.dilate(img, kernel_010)
        out_img = cv2.dilate(out_img, kernel_010_r)
        out_img = cv2.dilate(out_img, kernel_diag)
        out_img = cv2.dilate(out_img, kernel_diag_flip)
        for i in range(0,numdilate-1):
            out_img = cv2.dilate(out_img, kernel_010)
            out_img = cv2.dilate(out_img, kernel_010_r)
            out_img = cv2.dilate(out_img, kernel_diag)
            out_img = cv2.dilate(out_img, kernel_diag_flip)

    #Continuous erosion. Clears up unwanted pixels which are not part of road
    if(numerode>0):
        out_img = cv2.erode(out_img, kernel_010)
        out_img = cv2.erode(out_img, kernel_010_r)
        out_img = cv2.erode(out_img, kernel_diag)
        out_img = cv2.erode(out_img, kernel_diag_flip)
        for i in range(0,numerode-1):
            out_img = cv2.erode(out_img, kernel_010)
            out_img = cv2.erode(out_img, kernel_010_r)
            out_img = cv2.erode(out_img, kernel_diag)
            out_img = cv2.erode(out_img, kernel_diag_flip)

    #Median filtering to sharpen the edges
    if(nummedian>0):
        out_img = cv2.medianBlur(out_img, mediansz)
        for i in range(0,numerode-1):
            out_img = cv2.medianBlur(out_img, mediansz)

    return out_img
