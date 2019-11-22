import numpy as np
import cv2 as cv
from skimage.feature import hog

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
    '''
    Slices an array into patches of a sliding window
    without creating another array.
    :param img: Input image of size (w x h)
    :param patch_shape: Shape for patches of size (m x n)
    :return: Tensor with patches.
    '''
    X, Y, a = img.shape
    x, y = patch_shape
    shape = (X - x + 1, Y - y + 1, x, y, a)
    X_str, Y_str, a_str = img.strides
    strides = (X_str, Y_str, X_str, Y_str, a_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

def gridify(array, cells):
    grid = [np.array_split(b, cells, axis=1) for b in np.array_split(array, cells, axis=0)]
    return [g for v in grid for g in v]

def compute_features(image, nbins=16, cells=3):
    dx = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=1)
    dy = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=1)
    mag, angle = cv.cartToPolar(dx, dy, angleInDegrees=True)
    mag_grid = gridify(mag, cells)
    angle_grid = gridify(angle, cells)
    feature = np.zeros((cells**2, nbins))
    for i in range(cells ** 2):
        feature[i] = np.histogram(angle_grid[i].flatten(), nbins, weights=mag_grid[i].flatten())[0]
        feature[i] /= (1e-4 + np.sqrt(feature[i].dot(feature[i])))
    feature = feature.flatten()
    return feature / np.linalg.norm(feature)

