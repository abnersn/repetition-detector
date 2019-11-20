import numpy as np

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
