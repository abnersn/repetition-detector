from time import time

import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib.patches import Rectangle
from scipy.stats import mode, wasserstein_distance, trim_mean
from skimage import filters, segmentation
from skimage.feature import local_binary_pattern
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import util

PATCH_SIZE=16
THRESHOLD_PERCENTILE=5
DISTANCE_PERCENTILE=50
ORB_FEATURES=200
PADDING=5
N_BINS = 12
N_CELLS = 3
IMAGE='samples/img13.jpg'

# Loads image
image = np.array(Image.open(IMAGE))
start = time()
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

# Gets a single patch from the most frequent feature
x, y = kps[labels == mode(labels)[0]][0].astype(int)
patch = edges[y - PATCH_SIZE // 2:y + PATCH_SIZE // 2, x - PATCH_SIZE // 2:x + PATCH_SIZE // 2]
patch = patch.flatten()

edges = np.pad(edges, PATCH_SIZE // 2)
patches = util.patchify(edges[:, :, None], (PATCH_SIZE, PATCH_SIZE))
patches = patches.reshape((patches.shape[0], patches.shape[1], -1))
patches = patches ^ patch
similarity_map = patches.sum(axis=2)
similarity_map = filters.gaussian(similarity_map)

# Normalizes and binarizes feature map
similarity_map = similarity_map / similarity_map.max()
threshold = np.percentile(similarity_map.flatten(), 100 - THRESHOLD_PERCENTILE)
bmap = (similarity_map > threshold).astype(np.uint8)

# Filters noise
filter_size = int(np.sqrt(np.prod(image.shape) * 5.0e-6))
filter_size = filter_size * 2 + 1
bmap = cv.medianBlur(bmap, filter_size)
bmap = cv.dilate(bmap, np.ones((filter_size + 2, filter_size + 2), np.uint8))

# Removes detections close to the border.
bmap[0:PATCH_SIZE, :] = 1
bmap[-PATCH_SIZE:-1, :] = 1
bmap[:, -PATCH_SIZE:-1] = 1
bmap[:, 0:PATCH_SIZE] = 1
bmap = segmentation.flood_fill(bmap, (0, 0), 0)

# Computes bounding boxes
_, _, stats, _ = cv.connectedComponentsWithStats(bmap)
stats = np.delete(stats, 0, 0)
n_components = stats.shape[0]

features = []
gray = np.pad(gray, PATCH_SIZE // 2)
for i in range(n_components):
    x, y, w, h, a = stats[i]
    x -= PADDING
    y -= PADDING
    w += PADDING * 2
    h += PADDING * 2
    a = w * h
    stats[i] = np.array([x,y,w,h, a])
    bbox = gray[y:y+h, x:x+h]
    f = util.compute_features(bbox, N_BINS, N_CELLS)
    features.append(f)

distances = np.zeros((n_components, n_components))
for i in range(n_components):
    for j in range(n_components):
        e = np.sqrt((features[i] - features[j]) ** 2).sum()
        a = min(stats[i, -1], stats[j, -1]) / max(stats[i, -1], stats[j, -1])
        d = wasserstein_distance(features[i], features[j])
        distances[i, j] = d * e / np.sqrt(a)

limiar = np.percentile(distances, DISTANCE_PERCENTILE)
matches = (distances <= limiar).sum(axis=1)

is_different_match = (matches == matches.min())

distance_sum = trim_mean(distances, 0.1, axis=1) * is_different_match
is_different_distance = distance_sum == distance_sum.max()
is_different = is_different_match & is_different_distance

end = time() - start
print('Total running time {}'.format(end))

fig, ax = plt.subplots()
ax.imshow(image)
for i in range(n_components):
    m = matches[i]
    if m != matches.min():
        continue
    color = 'r' if is_different[i] else 'b'
    x, y, w, h, a = stats[i]
    r = Rectangle((x, y), w, h, linewidth=1,edgecolor=color,facecolor='none')
    ax.add_patch(r)
    # ax.text(x, y, distance_sum[i])
plt.show()