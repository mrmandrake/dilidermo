import numpy as np
#import pandas as pd
import cv2
import os
import glob
import sys
import itertools
import matplotlib.pyplot as plt
from scipy import ndimage
import sys
#from skimage.segmentation import active_contour

class Preprocessing:
    def PrePro(img):
        cv2.imshow('immagine',img)
        image = img.copy()
        # blur image
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # Applying CLAHE to resolve uneven illumination
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        kernel = np.ones((11, 11), np.uint8)
        for i in range(image.shape[-1]):
            image[:, :, i] = cv2.morphologyEx(
            image[:, :, i],
            cv2.MORPH_CLOSE, kernel)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imshow('hsv', hsv_image)
        return (hsv_image)
                
    def RoiMask(img,Open):            #Funzione per cercare di disegnare i contorni del neo
        
        #img=cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        new_img=img.copy()
        gray= cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
        for i in range (0,5):
            new_img = cv2.medianBlur(gray, 27)
        cv2.imshow('Blurred', new_img)
        ret2,thresh2 = cv2.threshold(new_img,127,255,cv2.THRESH_BINARY_INV)
        ret, thresh = cv2.threshold(thresh2, 170, 180, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(new_img, contours,-1,(0,255,253),thickness=3)
        cv2.imshow('Snake', new_img)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        areas=[]
        cnt_sizes=sorted(contour_sizes)
        big = cnt_sizes[-1][0]
        for x in range(len(cnt_sizes)):
            if cnt_sizes[x][0]>=big-0.15*big and cnt_sizes[x][0] <= big+0.15*big:
                areas.append(cnt_sizes[x][1])        
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [areas[0][:,:]], -1, 255, -1)
        cv2.imshow('mask', mask)        


img=cv2.imread(sys.argv[1])
Open=Preprocessing.PrePro(img)
Preprocessing.RoiMask(img, Open)
cv2.waitKey(0)
