import cv2
import glob
import numpy as np

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

images = [(cv2.imread(im), im) for im in glob.glob("./data/*.png")]
# img = cv2.imread("./data/ga.png")
fts = []

for (img, name) in images:
    ft = get_feature(img)
    fts.append((ft, name))

testimg = "./testset/at.png"
test = cv2.imread(testimg)
testft = get_feature(test)

min_d = 999999999
m = 'hi'
for (ft, name) in fts:
    if len(ft) < len(testft):
        ft = np.pad(ft, (0, len(testft)-len(ft)), 'constant',
                constant_values=(0,))
    elif len(ft) > len(testft):
        testft = np.pad(testft, (0, len(ft)-len(testft)), 'constant',
                constant_values=(0,))

    dist = np.linalg.norm(ft - testft)
    if dist < min_d:
        min_d = dist
        m = name
    print(name, dist)
print(m, min_d)

cv2.imshow("data", cv2.imread(m))
cv2.imshow("test", cv2.imread(testimg))
cv2.waitKey(0)
cv2.destroyAllWindows()
