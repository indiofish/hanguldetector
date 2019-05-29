from features import get_profile
import numpy as np
import cv2
import glob
from multiprocessing import Pool

FONTS = ["nanumgodigL", "nanumgodigB", "NanumGothic", "JosunIlboMJ", "BareunBatangL",
"BareunBatangB", "10X10"]
FILE_PATH = "../tensorflow-hangul-recognition/image-data/hangul-images/"

THREADS = 1

def aux(entry):
    """aux function for pooling"""

    img, name, font, c = entry
    return (get_profile(img), name, font, c)



if __name__ == '__main__':
    images = []
    for font in FONTS:
        # name[-6] is the position of the hangeul character
        images += [(cv2.imread(name), name, font, name[-6]) for name in glob.glob(FILE_PATH + "*{}_*.jpeg".format(font))]

    p = Pool(THREADS)

    data = p.map(aux, images)

    np.save("dataset", data)
