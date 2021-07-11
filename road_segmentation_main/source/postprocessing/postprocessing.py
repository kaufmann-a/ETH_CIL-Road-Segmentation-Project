#!/usr/bin/env python3
# coding: utf8

"""
Main function for classical post processing calls
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

from source.postprocessing.morphology import erodedilate_v0, erodedilate_v1

def postprocess(img, morphparam):
    """
    Applied the morphological filters on input image. erodedilate_v0/erodedilate_v1 are possible options with different types of filters.
    Function supports erosion/dilation/median filtering

    morphparam.numdilate: number of dilations
    morphparam.numerode: number of erosions
    morphparam.kernelsz kernel: size of erosion/dilation filters

    morphparam.nummedian: number of time median filters are applied
    morphparam.mediansz: size of the median filter kernel
    """
    #out_img = erodedilate_v0(img, morphparam.numdilate, morphparam.numerode, morphparam.nummedian, morphparam.kernelsz, morphparam.mediansz)
    out_img = img
    return out_img
