from outertest import *

#import argparse
#import io
#import os
import cv2
import glob
from PIL import Image
import imageio
#import numpy as np

'''
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/output.txt')
DEFAULT_DATA_DIR = os.path.join(SCRIPT_PATH, '../image-data')

DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64

def processdata(label_file, data_dir):
'''
images = [(cv2.imread(im),im) for im in glob.glob("../image-data/data/*.jpeg")]

#for (img,name) in images:
img = images[0][0]
x = get_feature2(img)
left = x[0]
top = x[1]
right = x[2]
bottom = x[3]

print(left)
print(top)
print(bottom)
print(right)

#cv2.rectangle(img,(left,top),(right,bottom),(0,0,255))
cv2.imshow("draw",img)
cv2.waitKey(0)

    #print(ft.shape)
    #print(ft)
    #im = Image.fromarray(ft)
    #im.save(name)
    #imageio.imwrite(name,ft)
        






'''

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
'''