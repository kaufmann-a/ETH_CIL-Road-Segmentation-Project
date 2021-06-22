from source.postprocessing.morphology import erodedilate_v0, erodedilate_v1

def postprocess(img, morphparam):
    #out_img = erodedilate_v0(img, morphparam.numdilate, morphparam.numerode, morphparam.nummedian, morphparam.kernelsz, morphparam.mediansz)
    out_img = img
    return out_img
