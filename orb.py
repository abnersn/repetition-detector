from time import time

import numpy as np
import cv2 as cv
import sys
from PIL import Image
from matplotlib.patches import Rectangle
from scipy.ndimage import shift
from scipy.stats import mode
from skimage import filters
from skimage.feature import local_binary_pattern
from skimage.morphology import selem, reconstruction
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import util

PATCH_SIZES=[8, 16, 32]
THRESH_PERCENTILE=5
FEATURE_PERCENTILE=40
N_BINS=40
N_CELLS=4
N_FEATURES=4
ORB_FEATURES=500
IMAGE='samples/portinari.jpg'

image = np.array(Image.open(IMAGE))
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
edges = cv.Canny(gray, 255 // 2, 255).astype(bool)

# Kernels for filtering.
filter_size = int(np.sqrt(np.prod(image.shape) * 5.0e-6))
filter_size = filter_size * 2 + 1
kernel_s = cv.getStructuringElement(cv.MORPH_ELLIPSE, (filter_size, filter_size))
kernel_m = cv.getStructuringElement(cv.MORPH_ELLIPSE, (filter_size + 2, filter_size + 2))
kernel_b = cv.getStructuringElement(cv.MORPH_ELLIPSE, (filter_size + 4, filter_size + 4))

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

# Gets most frequent features
counts, _ = np.histogram(labels, bins=(labels.max() - labels.min()))
frequent_features = counts.argsort()[-N_FEATURES:]

# Gets patches from the most frequent feature
similarity_maps = []
for feature_index in frequent_features:
    for size in PATCH_SIZES:
        x, y = kps[labels == feature_index][0].astype(int)
        patch = edges[y - size // 2:y + size // 2, x - size // 2:x + size // 2]
        patch = patch.flatten()

        padded_edges = np.pad(edges, size // 2)[:, :, None]
        patches = util.patchify(padded_edges, (size, size))
        patches = patches.reshape((patches.shape[0], patches.shape[1], -1))
        patches = patches & patch
        similarity_map = patches.sum(axis=2)
        similarity_map = filters.gaussian(similarity_map)

        similarity_map = similarity_map / similarity_map.max()
        similarity_maps.append(similarity_map)

similarity_map = np.stack(similarity_maps, axis=2)
similarity_map = similarity_map.sum(axis=2)
similarity_map = similarity_map / similarity_map.max() * 255
similarity_map = similarity_map.astype(np.uint8)

# _, binary_similarity_map = cv.threshold(similarity_map, 0, 255, cv.THRESH_OTSU)
binary_similarity_map = cv.adaptiveThreshold(
    similarity_map,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    (min(image.shape[0:2]) // 8) * 2 + 1,
    0
)
binary_similarity_map = cv.erode(binary_similarity_map, kernel_s)
# binary_similarity_map = cv.medianBlur(binary_similarity_map, kernel_s.shape[0])
# plt.imshow(image)
plt.imshow(binary_similarity_map, cmap='hot')
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