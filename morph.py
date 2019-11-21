import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.morphology import reconstruction
import util

SIZE = 7
THRESH_PERCENTILE = 25
IMAGE = 'samples/img1.jpg'
FEATURE_PERCENTILE=50
N_BINS=40
N_CELLS=5

kernel_s = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel_m = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
kernel_b = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

image = np.array(Image.open(IMAGE))
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

dx = cv.Sobel(gray, cv.CV_32F, 1, 0)
dy = cv.Sobel(gray, cv.CV_32F, 0, 1)
gray = np.sqrt(dx ** 2 + dy ** 2)
i_o = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel_m)
i_oc = cv.morphologyEx(i_o, cv.MORPH_CLOSE, kernel_m)

i_e = cv.erode(gray, kernel_m)
i_obr = reconstruction(i_e, gray)
i_ocd = cv.dilate(i_obr, kernel_m)

complement = lambda x: abs(x - x.max())
res = reconstruction(complement(i_ocd), complement(i_obr))
res = complement(res)

res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel_b) ** 2
res = res / res.max() * 255
res = np.round(res).astype(np.uint8)

thresh_window_size = np.prod(gray.shape) * 5e-5
thresh_window_size = int(np.round(thresh_window_size) * 2 + 1)
binary = cv.adaptiveThreshold(res, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, thresh_window_size, 0)
binary = cv.medianBlur(binary, 5)

################

# Computes bounding boxes
_, _, stats, _ = cv.connectedComponentsWithStats(binary)
stats = np.delete(stats, 0, 0)
n_bboxes = stats.shape[0]

# Compute features
gray_blur = cv.GaussianBlur(gray, (3,3), 3)
features = np.zeros((stats.shape[0], N_BINS * N_CELLS**2), dtype=np.float32)
for i in range(n_bboxes):
    x, y, w, h, a = stats[i]
    bbox = gray_blur[y:y+h, x:x+h]
    features[i] = util.compute_features(bbox, N_BINS, N_CELLS)

distances = np.zeros((features.shape[0], features.shape[0]))
for i in range(n_bboxes):
    area_i = stats[i, -1]
    for j in range(n_bboxes):
        area_j = stats[j, -1]
        a = min(area_i, area_j) / max(area_i, area_j)
        if a < 0.2:
            distances[i, j] = float('inf')
        else:
            chi = abs(cv.compareHist(features[i], features[j], cv.HISTCMP_CHISQR))
            distances[i, j] = chi * a
limiar = np.percentile(distances, FEATURE_PERCENTILE)

distances = (distances <= limiar).sum(axis=1)

fig, ax = plt.subplots()
ax.imshow(image)
for i, stat in enumerate(stats):
    matches = distances[i]
    if matches > 0.05 * max(distances):
        continue
    x, y, w, h, a = stat
    r = Rectangle((x, y), w, h, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(r)
    ax.text(x, y, str(matches))

plt.show()