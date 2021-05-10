from source.postprocessing.morphology import erodedilateallones

def postprocess(img, morphparam):
    out_img = erodedilateallones(img, morphparam.numdilate, morphparam.numerode, morphparam.nummedian, morphparam.kernelsz, morphparam.mediansz)
    return out_img