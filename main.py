import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
from sklearn.metrics import pairwise_distances
import time
import pathlib

# Display images in a grid
def display_images(images, size=10):
  if type(images) == dict:
    images_labels = [k for k in images.keys()]
  else:
    images_labels = [k for k in range(len(images))]
  grid_size = int(np.ceil(np.sqrt(len(images_labels))))
  for i in range(grid_size):
    fig = plt.figure(figsize=(size, size))
    for j in range(grid_size):
      index = i * grid_size + j
      if index >= len(images_labels):
          break
      label = images_labels[index]
      image = images[label]
      ax = fig.add_subplot(grid_size, grid_size, index + 1)
      ax.imshow(image)
      ax.set_xticks([])
      ax.set_yticks([])
      if type(label) == str:
        ax.set_xlabel(label)
    plt.show()

plt.rcParams['image.cmap'] = 'gray'

# Computes Gabor filters
angles = [i / 5 for i in range(5)]
sigmas = [1, 2, 4]
lambdas = [2, 3]
size = 15

filters = []
for angle in angles:
  for sigma in sigmas:
    for l in lambdas:
      args = {
        "ksize": (size, size),
        "sigma": sigma * size / 20,
        "theta": angle * np.pi,
        "lambd": size / l,
        "gamma": 1
      }
      kernel = cv.getGaborKernel(**args)
      filters.append(kernel)

print(len(filters))
display_images(filters)

# Load image
image = cv.imread("samples/img1.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = cv.resize(image, (512, 512))

r = cv.normalize(image[:, :, 0], cv.NORM_MINMAX, dtype=cv.CV_32F)
g = cv.normalize(image[:, :, 1], cv.NORM_MINMAX, dtype=cv.CV_32F)
b = cv.normalize(image[:, :, 2], cv.NORM_MINMAX, dtype=cv.CV_32F)

# Computes color maps
CM = {
  'RG': (r - g) / 2,
  'BY': (r + g - 2*b) / 4,
  'I': (r + g + b) / 3,
  'S': (np.amax(image, axis=2) - np.amin(image, axis=2)) / 2
}

# Display Color Maps
display_images(CM)

plt.rcParams['image.cmap'] = 'plasma'

# Computes gabor filters for each CM
feature_maps = []
for color_map in CM.values():
  for kernel in filters:
    f_map = cv.filter2D(color_map, cv.CV_32F, kernel)
    feature_maps.append(f_map)
display_images(feature_maps[:8:])

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
display_images(s_maps[:4:])

plt.rcParams['image.cmap'] = 'plasma'

computed = np.stack(s_maps, axis=2)
final = np.amax(computed, axis=2)
display_images([final])

