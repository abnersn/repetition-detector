import numpy as np
import cv2 as cv
from PIL import Image
import gc
from cv2.cv2 import DescriptorMatcher
from matplotlib.patches import Rectangle
from scipy.spatial import distance
from scipy.spatial.distance import hamming
from scipy.stats import mode
from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.filters import rank
from skimage.morphology import selem
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_fast, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

WINDOW_SIZE=64
RADIUS=3

def windowed_histogram_similarity(image, selem, reference_hist):
    # Compute normalized windowed histogram feature vector for each pixel
    px_histograms = rank.windowed_histogram(image, selem, n_bins=256).astype(np.float32)

    # Reshape coin histogram to (1,1,N) for broadcast when we want to use it in
    # arithmetic operations with the windowed histograms from the image
    reference_hist = reference_hist.reshape((1, 1) + reference_hist.shape)

    # Compute Chi squared distance metric: sum((X-Y)^2 / (X+Y));
    # a measure of distance between histograms
    X = px_histograms
    Y = reference_hist

    frac = (X - Y) ** 2 / (X + Y + 1.0e-4)

    chi_sqr = 0.5 * np.sum(frac, axis=2)

    # Generate a similarity measure. It needs to be low when distance is high
    # and high when distance is low; taking the reciprocal will do this.
    # Chi squared will always be >= 0, add small value to prevent divide by 0.
    similarity = 1 / (chi_sqr + 1.0e-4)

    return similarity

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
image = cv.resize(image, (512, 512))

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

# Gets one feature bbox from largest cluster
ref = kps[labels == mode(labels)[0]][0]

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_lbp = local_binary_pattern(gray, RADIUS, 8*RADIUS).astype(np.uint8)

x_l = max(int(ref[1]) - WINDOW_SIZE // 2, 0)
x_r = min(int(ref[1]) + WINDOW_SIZE // 2, image.shape[0])
y_l = max(int(ref[0]) - WINDOW_SIZE // 2, 0)
y_r = min(int(ref[0]) + WINDOW_SIZE // 2, image.shape[1])

patch = image_lbp[x_l:x_r, y_l:y_r]

patch_descriptor = np.histogram(patch.ravel(), bins=256, range=(0, 255))[0]
patch_descriptor = patch_descriptor.astype(float) / patch_descriptor.sum()

selem = selem.disk(WINDOW_SIZE // 2)

sim = windowed_histogram_similarity(image_lbp, selem, patch_descriptor)

fig, ax = plt.subplots()
ax.imshow(image)
ax.imshow(sim, cmap='hot', alpha=0.5)
plt.show()













