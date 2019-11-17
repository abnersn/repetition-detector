import numpy as np
import cv2 as cv
from PIL import Image
from cv2.cv2 import DescriptorMatcher
from scipy.spatial.distance import hamming
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_fast, pairwise_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def hamming_pairwise_distances(X):
    """
    Computes hamming distances between each sample of X
    :param X: List of orb descriptors in decimal form.
    :return: Hamming distance between each descriptor.
    """
    # Unpacks bits for matrix a, forming a MxN matrix
    X_bin = np.unpackbits(X, axis=1)

    # Broadcasts the matrix to MxNx1 and 1xNxM so that each lines of the
    # first matrix can be XORd to each line of the second along the 3rd dimension.
    X_bin = X_bin[:, :, None] ^ X_bin.T[None, :, :]
    return X_bin.sum(axis=1)

# Loads image
image = np.array(Image.open('samples/portinari.jpg'))

# Computes features
extractor = cv.ORB_create(nfeatures=500)
kps = extractor.detect(image)
_, dsc = extractor.compute(image, kps)

kps = np.array([k.pt for k in kps])

# Computes distances between each descriptor
distances = hamming_pairwise_distances(dsc)

# Clusterize by DBSCAN
margin = np.quantile(distances, 0.06)
clusterizer = DBSCAN(eps=margin, metric='precomputed')
labels = clusterizer.fit_predict(distances)

# Gets features from largest cluster and ignore the rest
kps = kps[labels == mode(labels)[0]]
labels = labels[labels == mode(labels)[0]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(image)
ax.scatter(kps[:, 0], kps[:, 1], c=labels)
plt.show()