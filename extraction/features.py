import cv2
import glob
import numpy as np

SPLIT_SIZE = 4
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k + min(i,m):(i+1) * k+min(i+1, m)] for i in range(n))

def get_profile(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # swap 255 and 0 so that contour can work correctly
    # contour sees an white image in a black background
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

    # ignore inner box for chars like 'o'
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    features = []

    for c in contours:
        # M = cv2.moments(c)
        x,y,w,h = cv2.boundingRect(c)
        # img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        for chunk in split(range(y, y+h-1), SPLIT_SIZE):
            tmp = 0
            for i in chunk:
                cnt = 0
                for j in range(x, x + w-1):
                    if thresh[i][j] == 0:
                        cnt += 1
                    else:
                        break
                tmp += cnt

            features.append(tmp/h)

        for chunk in split(range(y, y+h-1), SPLIT_SIZE):
            tmp = 0
            for i in chunk:
                cnt = 0
                for j in range(x+w-1, x, -1):
                    if thresh[i][j] == 0:
                        cnt += 1
                    else:
                        break
                tmp += cnt

            features.append(tmp/h)

        for chunk in split(range(x, x+w-1), SPLIT_SIZE):
            tmp = 0
            for j in chunk:
                cnt = 0
                for i in range(y, y + h-1):
                    if thresh[i][j] == 0:
                        cnt += 1
                    else:
                        break
                tmp += cnt

            features.append(tmp/w)

        for chunk in split(range(x, x+w), SPLIT_SIZE):
            tmp = 0
            for j in chunk:
                cnt = 0
                for i in range(y+h-1, y, -1):
                    if thresh[i][j] == 0:
                        cnt += 1
                    else:
                        break
                tmp += cnt
            features.append(tmp/w)

    return np.array(features)
