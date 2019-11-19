import numpy as np
import cv2 as cv
from PIL import Image
from cv2.cv2 import DescriptorMatcher
from scipy.spatial.distance import hamming
from scipy.stats import mode
from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_fast, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

WINDOW_SIZE=16
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

def patchify(img, patch_shape):
    X, Y, a = img.shape
    x, y = patch_shape
    shape = (X - x + 1, Y - y + 1, x, y, a)
    X_str, Y_str, a_str = img.strides
    strides = (X_str, Y_str, X_str, Y_str, a_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

# Loads image
image = np.array(Image.open('samples/img2.jpg'))
image = cv.resize(image, (512, 512))
gray = img_as_ubyte(cv.cvtColor(image, cv.COLOR_RGB2GRAY))

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

# Gets a single patch from the most frequent feature
x, y = kps[labels == mode(labels)[0]][0].astype(int)
patch = gray[y-WINDOW_SIZE//2:y+WINDOW_SIZE//2, x-WINDOW_SIZE//2:x+WINDOW_SIZE//2]
patch = patch.flatten()
patch = (patch - patch.mean()) / patch.std()

p = patchify(gray[:,:,None], (WINDOW_SIZE, WINDOW_SIZE)).astype(np.float32)

p = p.reshape((p.shape[0], p.shape[1], -1))
p = p - p.mean(axis=2)[:,:,None]
p = p / (1.0e-4 + p.std(axis=2)[:,:,None])
p = p * patch
p = p.sum(axis=2)

plt.imshow(p)
plt.show()


# Computes distance between such descriptor and the others.