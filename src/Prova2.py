import numpy as np
import pandas as pd
import cv2
import os
import glob
import sys
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
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
                
    def Lines(img,Open):
        SnakeCoord=[]
        colour=(150,105,80)
        for x in range (Open.shape[0]):
            for y in range(Open.shape[1]):
                r,g,b=Open[x,y]
                if r >= colour[0]:
                    if g>=colour[1]:
                        if b>=colour[2]:
                            SnakeCoord=SnakeCoord+[x,y]
                    
        
        coord = []
        for item in SnakeCoord:
            coord.append(item)
        p1=[]
        p2=[]
        for x in range(0,len(coord),2):
            p1=p1+coord(x)
        for y in range(1, len(coord), 2):
            p2=p2+coord[y]
        
            contour=active_contour(Open, coord, alpha=0.8, beta=0.7, w_line= -0.7, w_edge=0.5,
                                gamma=0.02,
                                bc=None,
                                max_px_move=1.0,
                                max_iterations=200)




path=os.getcwd()
mystring=IMG_list.List(path)
processing=Preprocessing()
iterations=3
for names in mystring:
    img=cv2.imread(names)
    Open=Preprocessing.PrePro(img)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Preprocessing.Lines(img, Open)

