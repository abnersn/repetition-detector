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
image = np.array(Image.open('samples/portinari.jpg'))

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

# Gets features from largest cluster and ignore the rest
kps = kps[labels == mode(labels)[0]]
labels = labels[labels == mode(labels)[0]]

# Gets a single keypoint and the bounding box around it
main_point = kps[0]
x, y = main_point - 32
x, y = int(x), int(y)
image[y:y+64, x:x+64] *= 0

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.imshow(image)
ax.scatter(x[:, 0], x[:, 1],x[:, 2], c='blue')
print(sum(pca.explained_variance_ratio_))
plt.show()