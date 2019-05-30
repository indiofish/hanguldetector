from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np


pca = PCA(.99)

testimage = Image.open('./image-data/hangul-images/hangul_1.jpeg','r')
testdata = np.asarray(testimage)
testdata_norm = normalize(testdata)
testpca_data = pca.fit_transform(testdata_norm)
testsum = np.sum(testpca_data,axis=1)

mindist = 10000
minimg = 0

for item in range(15):
	im = Image.open('./image-data/hangul-images/hangul_{}.jpeg'.format(item+1),'r')
	data = np.asarray(im)
	data_norm = normalize(data)
	pca_data = pca.fit_transform(data_norm)
	sum2 = np.sum(pca_data,axis=1)
	dist = np.linalg.norm(testsum-sum2)
	print(dist)
	if dist<mindist:
		minimg = item+1
		mindist = dist

'''
im2 = Image.open('./image-data/hangul-images/hangul_3.jpeg','r')
data2 = np.asarray(im2)
data2_norm = normalize(data2)
pca_data2 = pca.fit_transform(data2_norm)
'''



