from time import time

import numpy as np
import cv2 as cv
from PIL import Image
from scipy.stats import mode
from skimage import filters
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from util import hamming_pairwise_distances, patchify

WINDOW_SIZE=16
PERCENTILE=5
BLUR=True
ORB_FEATURES=200
IMAGE='samples/img4.jpg'

start = time()
# Loads image
image = np.array(Image.open(IMAGE))
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
gray = cv.Canny(gray, 255//2, 255).astype(bool)

# Computes features
extractor = cv.ORB_create(nfeatures=ORB_FEATURES)
kps = extractor.detect(image)
_, dsc = extractor.compute(image, kps)

kps = np.array([k.pt for k in kps])

# Computes distances between each descriptor
distances = hamming_pairwise_distances(dsc)

# Clusterize by DBSCAN
margin = np.quantile(distances, 0.04)
clusterizer = DBSCAN(eps=margin, metric='precomputed')
labels = clusterizer.fit_predict(distances)

# Gets a single patch from the most frequent feature
x, y = kps[labels == mode(labels)[0]][0].astype(int)
patch = gray[y-WINDOW_SIZE//2:y+WINDOW_SIZE//2, x-WINDOW_SIZE//2:x+WINDOW_SIZE//2]
patch = patch.flatten()

patches = patchify(gray[:, :, None], (WINDOW_SIZE, WINDOW_SIZE))
patches = patches.reshape((patches.shape[0], patches.shape[1], -1))
patches = patches ^ patch
similarity_map = patches.sum(axis=2)

if BLUR:
    similarity_map = filters.gaussian(similarity_map)

# Normalizes and binarizes feature map
similarity_map = similarity_map / similarity_map.max()
threshold = np.percentile(similarity_map.flatten(), 100 - PERCENTILE)
binary_similarity_map = (similarity_map > threshold).astype(np.uint8)

# Filters noise
filter_size = int(np.sqrt(np.prod(image.shape) * 5.0e-6))
filter_size = filter_size * 2 + 1
binary_similarity_map = cv.medianBlur(binary_similarity_map, filter_size)
binary_similarity_map = cv.dilate(binary_similarity_map, np.ones((filter_size + 2, filter_size + 2), np.uint8))

end = time()

print('Total running time: {}'.format(end - start))

plt.figure(1)
plt.imshow(image)
plt.imshow(binary_similarity_map, cmap='hot', alpha=0.4)
plt.show()