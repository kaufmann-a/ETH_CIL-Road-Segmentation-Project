#!/usr/bin/env python3
# coding: utf8

"""
Builds a torch loss function from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"


import numpy as np
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks
import math
import cv2
import os
from tqdm import tqdm

class HoughTransforms():

    def get_hough_transform(self, image):
        """
        image: image of predicted roads (black-white, 1 channel)
        """

        # skeletonize
        skeleton = skeletonize(image/255).astype(np.uint8)

        # TODO: if skeleton is too thin, use dilations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilate = cv2.dilate(skeleton, kernel, iterations=2)

        # get hough transforms

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(skeleton, theta=tested_angles)

        # draw lines on out_image
        out_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # TODO: tune these parameters
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=25, min_angle=30, min_distance=30)):
            #cv2.LineSegmentDetector
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            a = math.cos(angle)
            b = math.sin(angle)
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(out_image, pt1, pt2, color=(255,255,255), thickness=15)

        return out_image

    def create_line_images(self, directory):
        directory = directory
        out_directory = ""
        out_directory = os.path.join(out_directory, "pred_lines")
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        loop = tqdm(os.listdir(directory))
        for filename in loop:
            if filename.endswith(".png"):
                image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
                out_image = self.get_hough_transform(image)
                cv2.imwrite(os.path.join(out_directory, filename), out_image)

if __name__ == '__main__':
    ht = HoughTransforms()
    ht.create_line_images(directory = "../../../../trainings/exp_lr/20210619-102350-unet_exp_base_50_100/prediction-20210621-213346/pred-masks-original")