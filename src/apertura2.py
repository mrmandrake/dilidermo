# Apertura di immagini multiple
import os
import sys
import cv2
import numpy
import numpy.ma
import matplotlib
from matplotlib import pyplot as plt

img_path = "/Users/manovella/Projects/dilidermo/img/V4_2017/01/03/00001927_00003817/00013004.bmp"#sys.argv[1]

# Calcolo istrogramma
def histogram(grey, mask=None):
        channel = [0]
        hist = cv2.calcHist([grey], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(hist)
        plt.show()
        return (hist)

# 
def show_result(grey, laplacian, sobelx, sobely):
        matplotlib.pyplot.subplot(2, 2, 1)
        matplotlib.pyplot.imshow(grey, cmap='gray')
        matplotlib.pyplot.title('Original')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.subplot(2, 2, 2)
        matplotlib.pyplot.imshow(laplacian, cmap='gray')
        matplotlib.pyplot.title('Laplacian')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.subplot(2, 2, 3)
        matplotlib.pyplot.imshow(sobelx, cmap='gray')
        matplotlib.pyplot.title('Sobel X')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.subplot(2, 2, 4)
        matplotlib.pyplot.imshow(sobely, cmap='gray')
        matplotlib.pyplot.title('Sobel Y')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.show()


# Funzione di preprocessing immagini (Calcolo gradienti)
def test_gradient(img, grey):
        laplacian64 = cv2.Laplacian(grey, cv2.CV_64F)
        laplacian = numpy.uint8(numpy.absolute(laplacian64))
        sobelx64 = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=5)
        sobelx = numpy.uint8(numpy.absolute(sobelx64))
        sobely64 = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=5)
        sobely = numpy.uint8(numpy.absolute(sobely64))
        show_result(grey, laplacian, sobelx, sobely)

try:
        img = cv2.imread(img_path)
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imshow("Maledetta immagine", img)
        histogram(grey)
        test_gradient(img, grey)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
except:
        print("Oops! ", sys.exc_info()[0], "occured.")
