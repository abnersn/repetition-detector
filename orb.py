from time import time

import numpy as np
import cv2 as cv
import sys
from PIL import Image
from matplotlib.patches import Rectangle
from scipy.ndimage import shift
from scipy.stats import mode
from skimage import filters
from skimage.feature import local_binary_pattern, match_template
from skimage.morphology import selem
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import util

WINDOW_SIZE=32
THRESH_PERCENTILE=5
FEATURE_PERCENTILE=40
BLUR=True
N_BINS=50
N_CELLS=4
ORB_FEATURES=200
IMAGE='samples/img1.jpg'

image = np.array(Image.open(IMAGE))
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
edges = cv.Canny(gray, 255 // 2, 255).astype(bool)

# Computes features
extractor = cv.ORB_create(nfeatures=ORB_FEATURES)
kps = extractor.detect(image)
_, dsc = extractor.compute(image, kps)

kps = np.array([k.pt for k in kps])

# Computes distances between each descriptor
distances = util.hamming_pairwise_distances(dsc)

# Clusterize by DBSCAN
margin = np.quantile(distances, 0.04)
clusterizer = DBSCAN(eps=margin, metric='precomputed')
labels = clusterizer.fit_predict(distances)

# plt.imshow(image)
# plt.scatter(kps[:, 0], kps[:, 1], labels, c='yellow')
# plt.show()

# Gets a single patch from the most frequent feature
x, y = kps[labels == mode(labels)[0]][0].astype(int)
patch = gray[y - WINDOW_SIZE // 2:y + WINDOW_SIZE // 2, x - WINDOW_SIZE // 2:x + WINDOW_SIZE // 2]
similarity_map = match_template(gray, patch)

if BLUR:
    similarity_map = filters.gaussian(similarity_map)

plt.imshow(similarity_map)
plt.show()

# Normalizes and binarizes feature map
similarity_map = similarity_map / similarity_map.max()
threshold = np.percentile(similarity_map.flatten(), 100 - THRESH_PERCENTILE)
binary_similarity_map = (similarity_map > threshold).astype(np.uint8)

# Filters noise
filter_size = int(np.sqrt(np.prod(image.shape) * 5.0e-6))
filter_size = filter_size * 2 + 1
binary_similarity_map = cv.medianBlur(binary_similarity_map, filter_size)
binary_similarity_map = cv.dilate(binary_similarity_map, np.ones((filter_size + 2, filter_size + 2), np.uint8))

binary_similarity_map = shift(binary_similarity_map, WINDOW_SIZE//2)
plt.figure(1)
plt.imshow(image)
plt.imshow(binary_similarity_map, cmap='hot', alpha=0.2)
plt.show()

# Computes bounding boxes
_, _, stats, _ = cv.connectedComponentsWithStats(binary_similarity_map)
stats = np.delete(stats, 0, 0)
features = np.zeros((stats.shape[0], N_BINS * N_CELLS**2), dtype=np.float32)
for i, stat in enumerate(stats):
    x, y, w, h, a = stat
    bbox = gray[y:y+h, x:x+h]
    features[i] = util.compute_features(bbox, N_BINS, N_CELLS)

distances = np.zeros((features.shape[0], features.shape[0]))
for i in range(features.shape[0]):
    for j in range(features.shape[0]):
        a = min(stats[i, -1], stats[j, -1]) / max(stats[i, -1], stats[j, -1])
        if a < 0.5:
            distances[i, j] = float('inf')
        else:
            chi = abs(cv.compareHist(features[i], features[j], cv.HISTCMP_CHISQR))
            distances[i, j] = chi * a
limiar = np.percentile(distances, FEATURE_PERCENTILE)

distances = (distances <= limiar).sum(axis=1)

# print(limiar, distances[90, 92])
# input()

# boxes = [90,92]
# for b in boxes:
#     fig, ax = plt.subplots()
#     f = features[b]
#     ax.bar(np.arange(len(f)), f)
#     plt.show()


fig, ax = plt.subplots()
ax.imshow(image)
for i, stat in enumerate(stats):
    matches = distances[i]
    if matches > 0.05 * max(distances):
        pass
    x, y, w, h, a = stat
    r = Rectangle((x, y), w, h, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(r)
    ax.text(x, y, str(matches))


# ax.imshow(binary_similarity_map, cmap='hot', alpha=0.4)
plt.show()