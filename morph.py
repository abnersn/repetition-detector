import numpy as np
import cv2 as cv
import sys
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import trim_mean
from skimage.feature import hog
from skimage.morphology import reconstruction
import util

IMAGE = 'samples/img6.jpg'

kernel_s = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel_m = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
kernel_b = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

image = np.array(Image.open(IMAGE))
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
gray = cv.medianBlur(gray, 3)

dx = cv.Sobel(gray, cv.CV_32F, 1, 0)
dy = cv.Sobel(gray, cv.CV_32F, 0, 1)
gray = np.sqrt(dx ** 2 + dy ** 2)

i_e = cv.erode(gray, kernel_m)
i_obr = reconstruction(i_e, gray)
i_ocd = cv.dilate(i_obr, kernel_m)

complement = lambda x: abs(x - x.max())
res = reconstruction(complement(i_ocd), complement(i_obr))
res = complement(res)

res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel_m)
res = res / res.max() * 255
res = np.round(res).astype(np.uint8)

thresh_window_size = np.min(image.shape[0:2]) // 10 * 2 + 1
binary = cv.adaptiveThreshold(
    res,
    255,
    cv.ADAPTIVE_THRESH_MEAN_C,
    cv.THRESH_BINARY,
    thresh_window_size,
    0
)
binary = cv.erode(binary, kernel_b)
binary = cv.medianBlur(binary, 7)
binary = cv.dilate(binary, kernel_b)
plt.imshow(binary)
plt.show()
sys.exit()

# Computes bounding boxes
_, _, stats, _ = cv.connectedComponentsWithStats(binary)
stats = np.delete(stats, 0, 0)
diffs = abs(stats[:, -1, None] - stats[:, -1, None].T)
limiar = np.percentile(diffs, 50)
diffs = diffs < limiar
diffs = diffs.sum(axis=1)
stats = stats[diffs < diffs.mean()]

fig, ax = plt.subplots()
ax.imshow(image)
for i, stat in enumerate(stats):
    x, y, w, h, a = stat
    r = Rectangle((x, y), w, h, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(r)

plt.show()