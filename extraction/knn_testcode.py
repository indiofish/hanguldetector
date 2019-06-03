from features import get_profile
import numpy as np
import cv2
import glob
import knn
from multiprocessing import Pool
import timeit


FONTS = [""]
TESTFONT = "10X10"
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
    # data = np.load('dataset.npy')


    start = timeit.default_timer()
    error = 0
    cnt = 0
    for name in glob.glob(FILE_PATH+"*{}_*.jpeg".format(TESTFONT)):
        cnt += 1
        testimg = cv2.imread(name)
        img_ft = get_profile(testimg)
        char = name[-6]

        neighbors = knn.get_neighbors(data, img_ft, 10)
        response = knn.response(neighbors)
        label, confidence = response
        # print(votes)
        if char != label:
            error += 1
        print('gt:', char, 'infer:', label, 'confidence:', confidence)

    print(((256-error)/256)*100)
    print("took: ", timeit.default_timer() - start)
