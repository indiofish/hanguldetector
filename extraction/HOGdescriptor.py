import numpy as np
import PIL
from PIL import Image
import copy
import os, glob
import matplotlib
import matplotlib.pyplot as plt


# a function to remap angles
# our 9 bin histogram would have angles from 0 t0 160 with a 20 degree gap, 0, 20, 40, 60, 80, 100, 120, 140, 160
# so if we encounter an angle with a degree 180 it would be remapped to value 0
# or an angle of 200 would be remapped to value 20 and so on...

def remap(angle):
    if (int(angle) >= 180):  # handling angles from 180 to 360
        return int(angle) % 180
    else:
        return int(angle)  # angles 161 to 179 had to be handled separately


# a function to get the distribution angles and their magnitude if our angle is not one of the bins in our histogram
# for example : an angle of 76 degrees need to be distributed between angles 60 and 80
# as 60 is 16 units to the left of 76, while 80 is only 4 units to the right of 76
# 80% of the magnitude goes to angle 80 while 20% of the magnitude goes to angle 60

def find_distribution_angles(angle, mag, histogram):
    angle = remap(angle)

    if (angle >= 161 and angle <= 179):
        min_ = 0
        max_ = 160
        if (angle > 170):
            ratio_max = (angle - max_) / 20
            ratio_min = (180 - angle) / 20
        elif (angle < 170):
            ratio_max = (180 - angle) / 20
            ratio_min = (angle - max_) / 20
        else:
            ratio_max = 0.5
            ratio_min = 0.5
    else:
        max_ = 0

        for key, value in histogram.items():
            if (angle < key):
                max_ = key
                break

        min_ = max_ - 20
        ratio_max = (angle - min_) / 20
        ratio_min = (max_ - angle) / 20

    max_mag = round(ratio_max * mag, 8)
    min_mag = round(ratio_min * mag, 8)

    return [min_, min_mag, max_, max_mag]


def init_histo(histogram):
    for i in range(0, 9):
        histogram[i * 20] = 0


def HOG(image):
    img = Image.open(image)  # open image
    img = img.resize((64, 128), PIL.Image.ANTIALIAS)  # resize image to size 64x128

    img = img.convert('L')  # convert image to greyscale
    pixels = np.asarray(img).copy()  # get pixels

    rows = pixels.shape[0]
    columns = pixels.shape[1]

    x_gradient = np.asarray(img).copy()
    y_gradient = np.asarray(img).copy()

    # calculating the horizontal gradient
    for x in range(rows):
        for y in range(columns):
            if (y == columns - 1):
                x_gradient[x][y] = 0  # if it is the last row add 0
            else:
                x_gradient[x][y] = abs(int(pixels[x][y + 1]) - int(pixels[x][y]))  # else use the formula

    # calculating the vertical gradient
    for x in range(rows):
        for y in range(columns):
            if (x == rows - 1):
                y_gradient[x][y] = 0  # if it is the last column add 0
            else:
                y_gradient[x][y] = abs(int(pixels[x + 1][y]) - int(pixels[x][y]))  # else use the formula

    # calculating magnitude
    mag = np.sqrt(np.multiply(x_gradient, x_gradient) + np.multiply(y_gradient, y_gradient))

    # creating an array to store our angles
    angles = np.zeros((rows, columns))

    # calculating the angles (orientation of gradients)
    for x in range(rows):
        for y in range(columns):
            if (x_gradient[x][y] == 0):  # we don't want to divide by 0 and if it happens, we'll use 1 instead
                dx = 1
                dy = y_gradient[x][y]
            else:
                dx = x_gradient[x][y]
                dy = y_gradient[x][y]
            angles[x][y] = np.round(np.arctan(dy / dx) * (180 / np.pi))  # theeta = tan^-1(dy/dx)

    # creating a patch of 8x8
    patch_8x8_angles = np.zeros((8, 8))
    patch_8x8_mag = np.zeros((8, 8))

    # an array to store 8x8 patches of angles and magnitude
    angle_patches = []
    mag_patches = []

    move_patch_left = 0
    move_patch_down = 0

    for l in range(0, 16):  # we have 16 patches at each column
        for k in range(0, 8):  # we have 8 patches at each row
            for i in range(0, 8):  # in a patch we have 8 pixels in each column
                for j in range(0, 8):  # we have 8 pixels in each row
                    patch_8x8_angles[i][j] = angles[i + move_patch_down][j + move_patch_left]
                    patch_8x8_mag[i][j] = mag[i + move_patch_down][j + move_patch_left]
            move_patch_left = move_patch_left + 8
            angle_patches.append(copy.deepcopy(patch_8x8_angles))
            mag_patches.append(copy.deepcopy(patch_8x8_mag))
        move_patch_down = move_patch_down + 8
        move_patch_left = 0

    # we now have an array of 8x8 patches of angles and 8x8 patches of magnitude of our image of size 64x128

    # create a histogram
    histogram = {}

    init_histo(histogram)

    # a 9 bin histogram [(0, 0), (0, 20), (0, 40), (0, 60), (0, 80), (0, 100), (0, 120), (0, 140), (0, 160)]
    # representing first value as the magnitude of an angle and the second value as the angle
    # for example : 0 degree = 0, 20 degree = 0....
    # this would only hold a histogram for a single patch while we need to create a histogram for all patches

    Histograms = []

    # iterating over patches

    for i in range(128):  # 128 patches in total
        for j in range(8):  # each having 8 columns
            for k in range(8):  # and 8 rows
                if (int(angle_patches[i][j][
                            k]) in histogram):  # if an angle is one of our angles in our histogram, add its magnitude in the histogram respectively
                    histogram[int(angle_patches[i][j][k])] = histogram[int(angle_patches[i][j][k])] + mag_patches[i][j][
                        k]
                else:
                    distribution_angles = find_distribution_angles(int(angle_patches[i][j][k]), mag_patches[i][j][k],
                                                                   histogram)
                    histogram[distribution_angles[0]] = histogram[distribution_angles[0]] + distribution_angles[1]
                    histogram[distribution_angles[2]] = histogram[distribution_angles[2]] + distribution_angles[3]
        Histograms.append(copy.deepcopy(histogram))
        init_histo(histogram)

    # now we need to do block normalization so that our descriptor is not sensitive to overall lighting
    # i.e convert our histograms to one block containing 4 histograms each
    # and then normalize them

    # an array to store blocks of histogram, this would store a 16x16 block of histograms
    histogram_block = []

    # this would store all the blocks
    Histogram_16x16_Blocks = []

    j = 0
    i = 1
    k = 0

    # moving over all histograms and creating blocks of 4
    while (j != 119):
        if (k == 7):
            j = j + 1
            i = i + 1
            k = 0
            pass
        else:
            histogram_block.append((Histograms[j]))
            histogram_block.append((Histograms[i]))
            histogram_block.append((Histograms[j + 8]))
            histogram_block.append((Histograms[i + 8]))
            Histogram_16x16_Blocks.append(copy.deepcopy(histogram_block))
            histogram_block = []
            j = j + 1
            i = i + 1
            k = k + 1

    init_histo(histogram)

    normalized_value = 0

    # iterating over 16x16 blocks and normalizing them
    for j, k in enumerate(Histogram_16x16_Blocks):
        for l in range(0, 4):
            for m in range(0, 9):
                histogram[m * 20] = histogram[m * 20] + k[l][m * 20]
        normalized_value = np.sqrt((histogram[0] * histogram[0]) +  # normalized value
                                   (histogram[20] * histogram[20]) +
                                   (histogram[40] * histogram[40]) +
                                   (histogram[60] * histogram[60]) +
                                   (histogram[80] * histogram[80]) +
                                   (histogram[100] * histogram[100]) +
                                   (histogram[120] * histogram[120]) +
                                   (histogram[140] * histogram[140]) +
                                   (histogram[160] * histogram[160]))

        for l in range(0, 4):
            for m in range(0, 9):
                if (normalized_value != 0):
                    k[l][m * 20] = k[l][m * 20] / normalized_value  # normalized vector
                else:
                    k[l][m * 20] = 0

        init_histo(histogram)

    feature_vector = []

    # converting blocks into one feature vector of size 3780.
    # Positions for 16x16 blocks = 7 horizontal x 15 vertical = 105
    # Each 16x16 block is one 36x1 vector so concatenating them all together
    # 36x105 = 3780
    for blocks in (Histogram_16x16_Blocks):
        for block in blocks:
            for i in range(len(block)):
                feature_vector.append(block[i * 20])

    return feature_vector


path_dir = "./data/*"
filenames = glob.glob(path_dir)
# print(filename)

filenames = [file for file in filenames if file.endswith("다.jpeg")]
print(filenames)


HoG_features = dict()

for file in filenames:
    HoG_features[file.split('_')[2]] = HOG("{}".format(file))

# for feature in features:
#     error = []
#     for i in range(len(feature)):
#         err_temp = feature0[i] - feature0[i]
#         # print(np.square(err_temp))
#         error.append(np.square(err_temp))

input_text = "./data\\hangul_14_HamchorongbatangR_다.jpeg"
print()
print("Actually, input_text is {}".format(input_text.split('_')[2]))
input_feature = HOG(input_text)

for font, feature in HoG_features.items():
    if feature == input_feature:
        print()
        print("The font you put in might be : {}".format(font))


# error = []
# for i in range(len(feature1)):
#     err_temp = feature1[i] - feature2[i]
#     error.append(np.square(err_temp))
#
# mse3 = np.sum(error)/len(error)
#
# print(mse1, mse2, mse3)
#
# answer = min(mse1, mse2, mse3)
#
# print(answer)

fig = plt.figure()
plt.plot(input_feature)
plt.show()
fig.savefig('histogram1.png')
