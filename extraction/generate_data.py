from features import get_profile
import numpy as np
import cv2
import glob
from multiprocessing import Pool

FONTS = [""] # if "", include every font
TESTFONT = "" # if "", don't exclude the font 
FILE_PATH = "../tensorflow-hangul-recognition/image-data/hangul-images/"

THREADS = 4

def aux(entry):
    """aux function for pooling"""

    img, name, font, c = entry
    return (get_profile(img), name, font, c)


if __name__ == '__main__':
    images = []
    for font in FONTS:
        # name[-6] is the position of the hangeul character
        for name in glob.glob(FILE_PATH+"*{}_*.jpeg".format(font)):
            # exclude the testfont
            if TESTFONT not in name:
                images += [(cv2.imread(name), name, font, name[-6])]

    p = Pool(THREADS)

    data = p.map(aux, images)
    np.save("dataset", data)
