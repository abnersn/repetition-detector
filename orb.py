import numpy as np
import cv2 as cv
from PIL import Image
from cv2.cv2 import DescriptorMatcher
from scipy.spatial.distance import hamming
from scipy.stats import mode
from skimage.feature import local_binary_pattern
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_fast, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

WINDOW_SIZE=32
RADIUS=3

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
image = np.array(Image.open('samples/img1.jpg'))

# Computes features
extractor = cv.ORB_create(nfeatures=500)
kps = extractor.detect(image)
_, dsc = extractor.compute(image, kps)

kps = np.array([k.pt for k in kps])

# Computes distances between each descriptor
distances = hamming_pairwise_distances(dsc)

# Clusterize by DBSCAN
margin = np.quantile(distances, 0.04)
clusterizer = DBSCAN(eps=margin, metric='precomputed')
labels = clusterizer.fit_predict(distances)

# Computes image LBP
image_lbp = local_binary_pattern(cv.cvtColor(image, cv.COLOR_BGR2GRAY), RADIUS, 8*RADIUS)
descriptors = np.zeros((len(kps), 256))
for i in range(len(kps)):
    kp = kps[i]
    x_l = max(int(kp[1]) - WINDOW_SIZE // 2, 0)
    x_r = min(int(kp[1]) + WINDOW_SIZE // 2, image.shape[0])
    y_l = max(int(kp[0]) - WINDOW_SIZE // 2, 0)
    y_r = min(int(kp[0]) + WINDOW_SIZE // 2, image.shape[1])
    patch = image_lbp[x_l:x_r, y_l:y_r]
    descriptors[i] = np.histogram(patch.ravel(), bins=256, range=(0, 255))[0]
descriptors = descriptors / descriptors.sum(axis=1)[:, None]

pca = PCA(n_components=10)
x = pca.fit_transform(descriptors)

# Gets a single descriptor from the most frequent feature
main_descriptor = x[labels == mode(labels)[0]][0]

gm = GaussianMixture(n_components=10)
y = gm.fit_predict(x)

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(image)
ax.scatter(kps[:, 0], kps[:, 1], c=y)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x[:, 0], x[:, 1], c=y)
ax.scatter([main_descriptor[0]], [main_descriptor[1]], c='red')

plt.show()