import cv2
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import pathlib

image = cv2.imread("samples/img1.jpg")

# Color maps
CM = []
print(image.shape)