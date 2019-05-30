import cv2
import glob
import numpy as np


SPLIT_SIZE = 3
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k + min(i,m):(i+1) * k+min(i+1, m)] for i in range(n))

def get_feature(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # swap 255 and 0 so that contour can work correctly
    # contour sees an white image in a black background
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

    # ignore inner box for chars like 'o'
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    features = np.array([])

    for c in contours:
        # M = cv2.moments(c)
        x,y,w,h = cv2.boundingRect(c)
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)

        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        left, right, top, bottom = 0, 0, 0, 0
        for i in range(y, y+h):
            # left edge
            # in the direction of right
            # count consecutive black pixels until the edge
            cnt = 0
            for j in range(x, x+w):
                if thresh[i][j] == 0:
                    cnt += 1
                else:
                    break
            left += cnt
        # normalize(?)
        left /= h

        for i in range(y, y+h):
            # right edge
            cnt = 0
            for j in range(x+w, x-1,-1):
                if thresh[i][j] == 0:
                    cnt += 1
                else:
                    break
            right += cnt
        right /= h

        for j in range(x, x+w):
            cnt = 0
            for i in range(y, y+h):
                if thresh[i][j] == 0:
                    cnt += 1
                else:
                    break
            # print(cnt)
            top += cnt
        top /= w

        for j in range(x, x+w-1):
            cnt = 0
            for i in range(y+h-1, y, -1):
                if thresh[i][j] == 0:
                    cnt += 1
                else:
                    break
            bottom += cnt
        bottom /= h

        features = np.append(features, np.array([left, top, right, bottom]))

    return features

def get_feature2(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # swap 255 and 0 so that contour can work correctly
    # contour sees an white image in a black background
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

    # ignore inner box for chars like 'o'
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    

    # get outermost box
    corners = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        corners.append((x,y,x+w,y+h))
    left_c = min(corners,key=lambda x: x[0])[0]
    top_c = min(corners,key=lambda x: x[1])[1]
    right_c = max(corners,key=lambda x: x[2])[2]
    bottom_c = max(corners,key=lambda x: x[3])[3]
    img = cv2.rectangle(img, (left_c,top_c), (right_c,bottom_c), (0,255,0), 1)
    w = right_c - left_c
    h = bottom_c - top_c


    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    features = []

    for chunk in split(range(top_c, bottom_c), SPLIT_SIZE):
        tmp = 0
        for i in chunk:
            cnt = 0
            for j in range(left_c, left_c + w//2):
                if thresh[i][j] == 0:
                    cnt += 1
                else:
                    break
            tmp += cnt

        features.append(tmp/h)

    for chunk in split(range(top_c, bottom_c), SPLIT_SIZE):
        tmp = 0
        for i in chunk:
            cnt = 0
            for j in range(right_c, left_c + w//2, -1):
                if thresh[i][j] == 0:
                    cnt += 1
                else:
                    break
            tmp += cnt

        features.append(tmp/h)

    for chunk in split(range(left_c, right_c), SPLIT_SIZE):
        tmp = 0
        for j in chunk:
            cnt = 0
            for i in range(top_c, top_c + h//2):
                if thresh[i][j] == 0:
                    cnt += 1
                else:
                    break
            tmp += cnt

        features.append(tmp/w)

    for chunk in split(range(left_c, right_c), SPLIT_SIZE):
        tmp = 0
        for j in chunk:
            cnt = 0
            for i in range(bottom_c, top_c + h//2, -1):
                if thresh[i][j] == 0:
                    cnt += 1
                else:
                    break
            tmp += cnt
        features.append(tmp/w)



    return np.array(features)

def main():
    images = [(cv2.imread(im), im) for im in glob.glob("./data/*.png")]
    fts = []

    for (img, name) in images:
        ft = get_feature2(img)
        fts.append((ft, name))

    testimg = "./testset/bigan.png"
    test = cv2.imread(testimg)
    testft = get_feature2(test)
    print(testimg, testft)
    # cv2.imshow("image", test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    min_d = 999999999
    m = 'hi'
    for (ft, name) in fts:
        # try:
        dist = np.linalg.norm(ft - testft)
        # except:
        # tmp_ft = get_feature2(cv2.imread(name))
        # tmp_testft = get_feature2(test)
        # dist = np.linalg.norm(tmp_ft - tmp_testft)
        # print(dist)

        if dist < min_d:
            min_d = dist
            m = name
            print(name, dist, ft)
    print(m, min_d, ft)

    cv2.imshow("data", cv2.imread(m))
    cv2.imshow("test", cv2.imread(testimg))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

