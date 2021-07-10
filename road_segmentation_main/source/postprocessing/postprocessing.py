"""
Postprocess function
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

from source.postprocessing.morphology import erodedilate_v0, erodedilate_v1

def postprocess(img, morphparam):
    #out_img = erodedilate_v0(img, morphparam.numdilate, morphparam.numerode, morphparam.nummedian, morphparam.kernelsz, morphparam.mediansz)
    out_img = img
    return out_img
