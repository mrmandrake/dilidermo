import numpy as np
import pandas as pd
import cv2
import os
import glob
import sys
import itertools
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import active_contour

class IMG_list:
    def __init__(self, path):
        self.path = path


    def List(path):
        img_list=[]
        img_paths=[]
        for r,d, f in os.walk(path):
            for names in d:
                if names.endswith('img'):
                    for root, dir, files in os.walk(names):
                        if dir !=[]:
                            os.chdir(os.path.join(str(path),str(names), str(dir[0])))
                            path=os.getcwd()             
                            for r,d,f in os.walk(path):
                                for files in f:
                                    if files.endswith('.bmp'):
                                        img_list.append(os.path.join(str(r),str(files)))
        return (img_list)

class Preprocessing:
    def PrePro(img):
        cv2.imshow('immagine',img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
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
        cv2.waitKey(0)
        return (hsv_image)
                
    def Lines(img,Open):            #Funzione per cercare di disegnare i contorni del neo
        
        #img=cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        new_img=img.copy()
        gray= cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
        for i in range (0,5):
            new_img = cv2.medianBlur(gray, 27)
        cv2.imshow('Blurred', new_img)
        cv2.waitKey(0)
        ret2,thresh2 = cv2.threshold(new_img,127,255,cv2.THRESH_BINARY_INV)
        ret, thresh = cv2.threshold(thresh2, 170, 180, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(new_img, contours,-1,(0,255,253),thickness=3)
        cv2.imshow('Snake', new_img)
        cv2.waitKey(0)
        # Isolate largest contour
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
        #cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

        #cv2.imshow('biggest_contour',np.uint32([contour_sizes]))
        cv2.waitKey(0)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        #Preprocessing.Mask(contours, hierarchy, img)

    #def Mask(contours,hierarchy,img):
        

    def Center(img):
        gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        maxval=30
        minvect=[]
        itery=[]
        iterx=[]
        it=1
        for x in range(gray.shape[0]):
            for y in range(gray.shape[1]):
                val=gray[x,y]
                if val <= maxval:
                    minvect= minvect+[[x,y]]
                    while gray[x,y+it] <= maxval+10:
                        it=it+1
                        itery=itery+[it]
                    it=1
                    while gray[x+it,y]<=maxval+10:
                        it=it+1
                        iterx=iterx+[it]
                    it=1
                if x > gray.shape[0] or y> gray.shape[1]:
                    x=gray.shape[1]
                    y=gray.shape[0]
        iterx=max(iterx)
        itery=max(itery)
        if iterx>itery:
            radius=int(iterx)
        else:
            radius=int(itery)

        center=(minvect)

        return(center, radius)
    def Cerchio(resolution, center, radius):
        radians = np.linspace(0, 2*np.pi, resolution)
        c = center[1] + radius*np.cos(radians)#polar co-ordinates
        r = center[0] + radius*np.sin(radians)
    
        return np.array([c, r]).T

  








path=os.getcwd()
mystring=IMG_list.List(path)
processing=Preprocessing()
iterations=3
for names in mystring:
    img=cv2.imread(names)
    Open=Preprocessing.PrePro(img)
    Preprocessing.Lines(img, Open)
    #centro=Preprocessing.Center(img)
    #for u in range(len(centro[0])):
    #    punti=Preprocessing.Cerchio(1000,centro[0][u],centro[1])[:-1]
     #   snake = active_contour(img, punti)
      #  cv2.polylines(img, np.int32([snake]), True, (0,255,255), 3)
       # cv2.imshow('Snake', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()