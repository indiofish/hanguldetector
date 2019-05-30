import cv2
import numpy as np
import features
import score

testimgpath = "./testset/testga.png"
img = cv2.imread(testimgpath)
img_ft = features.get_profile(img)

min_loss = 99999999 # some magic big number
min_name = ''
# cv2.destroyAllWindows()

data = np.load('./dataset.npy')

for (feature, fullname, font, c) in data:
    loss = score.loss(feature, img_ft)
    if loss < min_loss:
        min_loss = loss
        min_name = fullname

cv2.imshow("data", cv2.imread(min_name))
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
