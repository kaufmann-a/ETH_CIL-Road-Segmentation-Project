from source.postprocessing.morphology import erodedilate_v0, erodedilate_v1

def postprocess(img, morphparam):
    out_img = erodedilate_v1(img, morphparam.numdilate, morphparam.numerode, morphparam.nummedian, morphparam.kernelsz, morphparam.mediansz)
    return out_img