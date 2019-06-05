from PIL import Image
import os
import cv2
from tesserocr import PyTessBaseAPI, RIL
import numpy as np
import matplotlib.pyplot as plt


def get_image(name):
    path = os.path.dirname(os.path.realpath(__file__)) + name
    #path = '/home/wonho/ImageParsing/resource/orc_ori_image/etc1.JPG'
    image = Image.open(path).convert('L')
    return image

def split_image(image, output_size=64):
    # image = get_image(name)
    splitted = []

    with PyTessBaseAPI(lang='eng+kor') as api:
        api.SetImage(image)
        boxes = api.GetComponentImages(RIL.TEXTLINE, False)

        pixels = image.load()
        histograms = []
        for i, (im, box, _, _) in enumerate(boxes):
            # im is a PIL image object
            # box is a dict with x, y, w and h keys
            api.SetRectangle(box['x'], box['y'], box['w'], box['h'])

            histogram = []
            for j in range(box['x'], box['x'] + box['w']):
                sum_pixel_value = 0
                for k in range(box['y'], box['y'] + box['h']):
                    sum_pixel_value += pixels[j, k]
                histogram.append(sum_pixel_value // box['h'])

            histograms.append(histogram)

            threshold = max(histogram) - 2
            starts = []
            ends = []
            for j in range(len(histogram)):
                if j == 0 or (histogram[j] <= threshold < histogram[j-1]):
                    starts.append(j)
                if j == len(histograms[i])-1 or (histogram[j] <= threshold < histogram[j+1]):
                    ends.append(j)

            for j in range(len(starts)):
                if j >= len(starts):
                    break
                if ends[j] - starts[j] < box['h'] * 0.5:
                    if j > 0:
                        del starts[j]
                        del ends[j-1]

            cuts = []
            for j in range(len(starts)):
                if j == 0:
                    cuts.append(starts[j])
                else:
                    cuts.append((starts[j] + ends[j-1]) // 2)

            if (len(ends) > 0):
                cuts.append(ends[-1])

            for j in range(len(cuts)-1):
                img = np.array(image.crop((box['x'] + cuts[j], box['y'], box['x'] + cuts[j+1], box['y']+box['h'])))

                # swap 255 and 0 so that contour can work correctly
                # contour sees an white image in a black background
                ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

                # ignore inner box for chars like 'o'
                tmp =  cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # based on opencv version, number of parameters differ
                try:
                    _, contours, hierarchy = tmp
                except:
                    contours, hierarchy = tmp

                # get outermost box
                corners = []
                for c in contours:
                    x,y,w,h = cv2.boundingRect(c)
                    corners.append((x,y,x+w,y+h))
                left_c = min(corners,key=lambda x: x[0])[0]
                top_c = min(corners,key=lambda x: x[1])[1]
                right_c = max(corners,key=lambda x: x[2])[2]
                bottom_c = max(corners,key=lambda x: x[3])[3]

                cropped = img[top_c:bottom_c+1, left_c: right_c+1]
                resized = cv2.resize(cropped,(output_size, output_size))
                splitted.append(resized)

    return splitted



if __name__ == "__main__":
    path = os.path.dirname(os.path.realpath(__file__)) + "/testset/Test3.png"
    #path = '/home/wonho/ImageParsing/resource/orc_ori_image/etc1.JPG'
    image = Image.open(path).convert('L')
    with PyTessBaseAPI(lang='eng+kor') as api:
        api.SetImage(image)
        boxes = api.GetComponentImages(RIL.TEXTLINE, False)
        print('Found {} textline image components.'.format(len(boxes)))

        pixels = image.load()
        histograms = []
        count = 0
        for i, (im, box, _, _) in enumerate(boxes):
            # im is a PIL image object
            # box is a dict with x, y, w and h keys
            api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
            ocrResult = api.GetUTF8Text()
            conf = api.MeanTextConf()

            histogram = []
            for j in range(box['x'], box['x'] + box['w']):
                sum_pixel_value = 0
                for k in range(box['y'], box['y'] + box['h']):
                    sum_pixel_value += pixels[j, k]
                histogram.append(sum_pixel_value // box['h'])

            histograms.append(histogram)

            print("Box[{0}]: x={x}, y={y}, w={w}, h={h}, confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))

            threshold = max(histogram) - 2
            starts = []
            ends = []
            for j in range(len(histogram)):
                if j == 0 or (histogram[j-1] > threshold and histogram[j] <= threshold):
                    starts.append(j)
                if j == len(histograms[i])-1 or (histogram[j] <= threshold and histogram[j+1] > threshold):
                    ends.append(j)

            for j in range(len(starts)):
                if j >= len(starts):
                    break
                if ends[j] - starts[j] < box['h'] * 0.5:
                    if j > 0:
                        del starts[j]
                        del ends[j-1]

            cuts = []
            for j in range(len(starts)):
                if j == 0:
                    cuts.append(starts[j])
                else:
                    cuts.append((starts[j] + ends[j-1]) // 2)

            if (len(ends) > 0):
                cuts.append(ends[-1])
            print(cuts)

            for j in range(len(cuts)-1):
                img = np.array(image.crop((box['x'] + cuts[j], box['y'], box['x'] + cuts[j+1], box['y']+box['h'])))
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # swap 255 and 0 so that contour can work correctly
                # contour sees an white image in a black background
                ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

                # ignore inner box for chars like 'o'
                tmp =  cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # based on opencv version, number of parameters differ
                try:
                    _, contours, hierarchy = tmp
                except:
                    contours, hierarchy = tmp
                print('contours' + str(len(contours)))



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

                cv2.imshow("image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                filename = 'result{}.png'.format(count)
                cv2.imwrite(filename, img)
                Image.open(filename).crop((left_c, top_c, right_c, bottom_c)).resize((64,64)).save(filename)

                count = count + 1
