import os, sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
from sklearn.metrics import pairwise_distances
import time
import pathlib

# Computes Gabor filters
angles = [i / 5 for i in range(5)]
size = 3

filters = []
filter_names = []
for angle in angles:
  args = {
    "ksize": (size, size),
    "sigma": 0.75,
    "theta": angle * np.pi,
    "lambd": 7.5,
    "gamma": 1
  }
  kernel = cv.getGaborKernel(**args)
  filters.append(kernel)
  filter_names.append('t = {}'.format(angle))

# Load image
image = cv.imread("samples/portinari.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

r = cv.normalize(image[:, :, 0], cv.NORM_MINMAX, dtype=cv.CV_32F)
g = cv.normalize(image[:, :, 1], cv.NORM_MINMAX, dtype=cv.CV_32F)
b = cv.normalize(image[:, :, 2], cv.NORM_MINMAX, dtype=cv.CV_32F)

# Computes color maps
color_maps = np.stack([
  (r - g) / 2,
  (r + g - 2*b) / 4,
  (r + g + b) / 3,
  (np.amax(image, axis=2) - np.amin(image, axis=2)) / 2
], axis=2).astype(np.float32)

# Computes feature maps
feature_maps = []
for kernel in filters:
  maps = cv.filter2D(color_maps, cv.CV_32F, kernel)
  feature_maps.append(maps)
feature_maps = np.concatenate(feature_maps, axis=2)

print(feature_maps.shape)

for i in range(feature_maps.shape[-1]):
  norm = cv.normalize(feature_maps[:, :, i], cv.NORM_MINMAX)
  norm = (norm > 0.1 * norm.max()).astype(np.uint8) * 255
  cv.imshow('image', norm)
  k = cv.waitKey(0)
  print(filter_names[i % len(filters)], k)

sys.exit()

def compute_roundness(patch):
  contours = cv.findContours(patch, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[1][0].squeeze()
  if len(contours.shape) != 2:
    return 0
  center = contours.mean(axis=0)
  distances = pairwise_distances(contours, [center])
  if distances.max() == 0:
    return 0
  return distances.min() / distances.max()

def process_feature_map(f_map):
  result = np.copy(f_map)
  result = (result - result.min()) / (result.max() - result.min())
  result = result.astype(np.float32)

  binary = (result > 0.6 * result.max()).astype(np.uint8)

  # Gets connected components
  _, labels, stats, centroids = cv.connectedComponentsWithStats(binary)
  
  # Skips empty feature maps
  if len(stats) == 0:
    return result
  
  # Deletes background stats
  stats = np.delete(stats, 0, 0)
  centroids = np.delete(centroids, 0, 0)

  # Global FM inhibition
  global_inhibition = (len(stats) * binary.sum())

  # Total area
  total_area = np.prod(result.shape)

  # Fix switched x and y
  centroids[:, 0], centroids[:, 1] = centroids[:, 1], centroids[:, 0]
  
  # Compute distances factors
  C = pairwise_distances(centroids)
  C = np.exp(-C**2/C.var())

  # Compute area factors
  A = stats[:, -1]
  A = (A - A.mean()) / A.std()
  
  # Computes roundess and max feature map value
  max_responses = []
  roundness_values = []
  for j in range(len(stats)):
    y, x, h, w, a = stats[j]
    feature_patch = result[x:x+w, y:y+h] 
    binary_patch = binary[x:x+w, y:y+h]
    roundness_values.append(compute_roundness(binary_patch))
    max_responses.append(feature_patch.max())
  max_responses = np.array(max_responses)
  roundness_values = np.array(roundness_values)

  # Computes inhibitions
  binary = binary.astype(np.float32)
  for j in range(len(stats)):
    y, x, h, w, r = stats[j]

    r = roundness_values[j]
    a = a * max_responses[j]
    c = C[j] * max_responses
    c[j] = 0
    c = c.mean()

    local_inhibition = r * (max_responses[j] - (a + c))

    binary[x:x+w, y:y+h] *= local_inhibition
  binary_gauss = cv.GaussianBlur(binary, (5, 5), 5)
  limiar_gauss = cv.normalize(binary_gauss, 0, 1, cv.NORM_MINMAX)
  result *= global_inhibition
  result *= 10 * limiar_gauss

  return result

s_maps = list(map(process_feature_map, feature_maps))

computed = np.stack(s_maps, axis=2)
final = np.amax(computed, axis=2)
display_images([final])

