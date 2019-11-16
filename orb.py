import numpy as np
import cv2 as cv
from PIL import Image
from cv2.cv2 import DescriptorMatcher
from scipy.spatial.distance import hamming
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_fast, pairwise_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loads image
image = np.array(Image.open('samples/img1.jpg'))

# Computes features
extractor = cv.ORB_create(nfeatures=2000)
kps = extractor.detect(image)
kps = np.array([k.pt for k in kps])

# Compute distances
distances = pairwise_distances(kps, kps)

# Clusterizes
margin = np.quantile(distances, 0.005)
clusterizer = DBSCAN(eps=margin, metric='precomputed')
labels = clusterizer.fit_predict(distances)

# Compute cluster centroids
centroids = []
for label in np.unique(labels):
    if label == -1:
        continue
    cluster = kps[labels == label]
    centroids.append(cluster.mean(axis=0))
centroids = np.array(centroids)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(image)
ax.scatter(centroids[:, 0], centroids[:, 1])
plt.show()