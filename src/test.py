import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import matplotlib as mp
from matplotlib import pyplot

# commento di prova

def find_biggest_contour(image):
    # Copy
    image = image.copy()
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour)
                     for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

    pyplot.subplot(1, 1, 1)
    pyplot.imshow(mask)
    pyplot.show()

def thresholds_test(img):
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY',
    'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        pyplot.subplot(2, 3, i+1)
        pyplot.imshow(images[i], 'gray')
        pyplot.title(titles[i])
        pyplot.xticks([]),
        pyplot.yticks([])

    pyplot.show()

app = QApplication(sys.argv)
image_path = QFileDialog.getOpenFileName()[0]
img = cv2.imread(image_path, 0)
find_biggest_contour(img)
thresholds_test(img)
sys.exit(app.exec_())