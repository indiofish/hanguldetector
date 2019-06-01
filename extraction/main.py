import cv2
from HoG import calculate_hog_features
import numpy as np
import features
import score
from visualize_data import *
import timeit

if __name__ == '__main__':
    img = plt.imread("./data/an3.png")

    hist_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])

    start = timeit.default_timer()
    hist_cells, hog_feature = calculate_hog_features(img, hist_bins)
    print(timeit.default_timer() - start)

    # visualize_histogram_all_cells(hist_cells, hist_bins)

    visualize_image(hog_feature)

    testimgpath = "./testset/testda.png"
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
        print(c, loss)

    print(min_name, min_loss)

    cv2.imshow("data", cv2.imread(min_name))
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
