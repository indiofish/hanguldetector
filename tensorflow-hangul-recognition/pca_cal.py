from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from PIL import Image
from outertest import *
import numpy as np
import cv2
import glob



pca = PCA(.99)
mindist = 0.1
minimg = 0

#testimage = Image.open('./image-data/hangul-images/hangul_1.jpeg','r')
images = [(cv2.imread(im), im) for im in glob.glob("./image-data/data/*.jpeg")]
#fts = []

testimage = cv2.imread('./image-data/data/hangul_44.jpeg')
ft = get_feature2(testimage)
#im = Image.fromarray(ft)
testdata = np.reshape(ft,(-1,2))
testdata_norm = normalize(testdata)
testpca_data = pca.fit_transform(testdata_norm)
testsum = np.sum(testpca_data,axis=1)


for (img, name) in images:
	ft = get_feature2(img)
	nameFp = name.split('_')[-1].split('.')[0]
	data = np.reshape(ft,(-1,2))
	data_norm = normalize(data)
	pca_data = pca.fit_transform(data_norm)
	sum2 = np.sum(pca_data,axis=1)
	dist = np.linalg.norm(testsum-sum2)
	print(nameFp,end=" ")
	print(dist)
	if dist<mindist:
		print(nameFp+"WWWW")
		

	#fts.append((im, name))




'''

for item in range(110):
	im = Image.open('./image-data/hangul-images/hangul_{}.jpeg'.format(item+1),'r')
	data = np.asarray(im)
	data_norm = normalize(data)
	pca_data = pca.fit_transform(data_norm)
	sum2 = np.sum(pca_data,axis=1)
	dist = np.linalg.norm(testsum-sum2)
	print(item+1,end=" ")
	print(dist)
	if dist<mindist:
		minimg = item+1
		mindist = dist

im2 = Image.open('./image-data/hangul-images/hangul_3.jpeg','r')
data2 = np.asarray(im2)
data2_norm = normalize(data2)
pca_data2 = pca.fit_transform(data2_norm)
'''



