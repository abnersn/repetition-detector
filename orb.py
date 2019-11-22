from time import time

import numpy as np
import cv2 as cv
import sys

import skimage
from PIL import Image
from matplotlib.patches import Rectangle
from scipy.stats import trim_mean
from skimage import filters, morphology, segmentation
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import util

PATCH_SIZES=[8, 16]
AREA_DIFF_LIMIAR=0.9
FEATURE_PERCENTILE=40
N_BINS=8
N_CELLS=2
N_FEATURES=5
ORB_FEATURES=500
IMAGE='samples/img6.jpg'

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

# Computes distance matrix between descriptors
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
    features_cluster = kps[labels == feature_index]
    for size in PATCH_SIZES:
        # Gets one sample from the cluster
        x, y = features_cluster[0].astype(int)
        patch = edges[y - size // 2:y + size // 2, x - size // 2:x + size // 2]
        patch = patch.flatten()

        # Computes similarity map
        padded_edges = np.pad(edges, size // 2)[:, :, None]
        patches = util.patchify(padded_edges, (size, size))
        patches = patches.reshape((patches.shape[0], patches.shape[1], -1))
        patches = patches & patch
        similarity_map = patches.sum(axis=2)

        # A gaussian filter with window size will highlight peaks even more
        similarity_map = filters.gaussian(similarity_map, size / 2)

        # Weights map by the number of elements in the cluster
        similarity_map = similarity_map * features_cluster.shape[0]

        similarity_maps.append(similarity_map)

# Stacks map, sum and smooth
similarity_map = np.stack(similarity_maps, axis=2)
similarity_map = similarity_map.sum(axis=2)
similarity_map = (similarity_map / similarity_map.max()) ** 12
similarity_map = similarity_map / similarity_map.max() * 255
similarity_map = similarity_map.astype(np.uint8)

# Thresholds response map
limiar = skimage.filters.threshold_triangle(similarity_map)
bmap = similarity_map > limiar
bmap = morphology.area_opening(bmap.astype(int), 12).astype(np.uint8)
border_width = int(sum(PATCH_SIZES) * 1.5)

bmap[0:PATCH_SIZES[-1], :] = 1
bmap[-PATCH_SIZES[-1]:-1, :] = 1
bmap[:, -PATCH_SIZES[-1]:-1] = 1
bmap[:, 0:PATCH_SIZES[-1]] = 1
bmap = segmentation.flood_fill(bmap, (0, 0), 0)

# Computes bounding boxes
_, _, stats, _ = cv.connectedComponentsWithStats(bmap)
stats = np.delete(stats, 0, 0)

# Clusterize areas
areas = stats[:, -1]
limiar = trim_mean(areas, 0.3)
area_diffs = abs(areas - limiar) / limiar

# Removes outliers by area
stats = stats[area_diffs <= AREA_DIFF_LIMIAR, :]
area_diffs = area_diffs[area_diffs <= AREA_DIFF_LIMIAR]

features_cluster = np.zeros((stats.shape[0], N_BINS * N_CELLS ** 2), dtype=np.float32)
for i, stat in enumerate(stats):
    x, y, w, h, a = stat
    bbox = gray[y:y+h, x:x+h]
    features_cluster[i] = util.compute_features(bbox, N_BINS, N_CELLS)

distances = np.zeros((features_cluster.shape[0], features_cluster.shape[0]))
for i in range(features_cluster.shape[0]):
    for j in range(features_cluster.shape[0]):
        a = min(stats[i, -1], stats[j, -1]) / max(stats[i, -1], stats[j, -1])
        if a < 0.5:
            distances[i, j] = float('inf')
        else:
            chi = abs(cv.compareHist(features_cluster[i], features_cluster[j], cv.HISTCMP_CHISQR))
            distances[i, j] = chi
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

#
fig, ax = plt.subplots()
ax.imshow(image)
for i, stat in enumerate(stats):
    matches = distances[i]
    x, y, w, h, a = stat
    r = Rectangle((x, y), w, h, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(r)
    ax.text(x, y, str(matches))

plt.show()