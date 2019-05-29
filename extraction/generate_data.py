from features import get_profile
import numpy as np
import cv2
import glob

FONTS = ["NanumGothic", "JosunIlboMJ", "BareunBatangL", "10X10"]
FILE_PATH = "../tensorflow-hangul-recognition/image-data/hangul-images/"

images = []
for font in FONTS:
    # name[-6] is the position of the hangeul character
    images += [(cv2.imread(name), name, font, name[-6]) for name in glob.glob(FILE_PATH + "*{}_*.jpeg".format(font))]

data = []
for (img, name, font, c) in images:
    profile = get_profile(img)
    data.append((np.array(profile), name, font, c))

np.save("dataset", data)
