from features import get_profile
import numpy as np
import cv2
import glob
import knn
from multiprocessing import Pool
import timeit


FONTS = [""]
TESTFONT = "BareunDotum1"
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

    start = timeit.default_timer()
    error = 0
    for name in glob.glob(FILE_PATH+"*{}_*.jpeg".format(TESTFONT)):
        testimg = cv2.imread(name)
        img_ft = get_profile(testimg)
        char = name[-6]

        neighbors = knn.get_neighbors(data, img_ft, 10)
        response = knn.response(neighbors)
        # print(votes)
        if char != response:
            error += 1
            print('gt:', char, 'infer:', response)

    print(((256-error)/256)*100)
    print("took: ", timeit.default_timer() - start)
