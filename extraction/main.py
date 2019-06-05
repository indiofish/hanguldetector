from features import get_profile
import image_parser
import numpy as np
import cv2
import glob
import knn
import timeit
from collections import namedtuple

TESTIMG = "/testset/Test3.png"
DATA = 'dataset.npy'
DISPLAY = 1

if __name__ == '__main__':
    data = np.load(DATA)
    img = image_parser.get_image(TESTIMG)
    splitted = image_parser.split_image(img)

    lst = []
    tmp = namedtuple("tmp", "img label confidence")
    for char_img in splitted:
        ft = get_profile(char_img)

        neighbors = knn.get_neighbors(data, ft, 10)
        response = knn.response(neighbors)
        label, confidence = response

        lst.append(tmp(char_img, label, confidence))

    for t in lst:
        char_img = t.img
        label = t.label
        confidence = t.confidence
        if DISPLAY:
            ret = "infer: {}, confidence{}".format(label, confidence)
            print(ret)
            cv2.imshow(ret, char_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
