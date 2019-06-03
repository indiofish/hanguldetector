from outertest import *

import argparse
import io
import os
import cv2
import glob
from PIL import Image
import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/output.txt')
DEFAULT_DATA_DIR = os.path.join(SCRIPT_PATH, '../image-data')

DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64

def processdata(label_file, data_dir):

    images = [(cv2.imread(im),im) for im in glob.glob("../image-data/data/*.jpeg")]

    for (img,name) in images:
        ft = get_feature2(img)
        print(ft.shape)








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--data-dir', type=str, dest='data_dir',
                        default=DEFAULT_DATA_DIR,
                        help='Directory of database to modify.')
    args = parser.parse_args()
    processdata(args.label_file, args.data_dir)
