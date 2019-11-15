import numpy as np
import pandas as pd
import cv2
import os
import glob
import sys
import itertools
import matplotlib.pyplot as plt
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
                
    def Lines(img,Open):            #Funzione per cercare di disegnare i contorni del neo
        SnakeCoord=[]
        colour=(130,50,120)             #inizializzazione delle coordinate di uno snake random in
        for x in range (Open.shape[0]): #modo tale che abbia senso la sua inizializzazione (basata quindi su una treshold di colore)
            for y in range(Open.shape[1]):
                r,g,b=Open[x,y]
                if r >= colour[0]:
                    if g>=colour[1]:
                        if b>=colour[2]:
                            SnakeCoord=SnakeCoord+[x,y] 
                    
        coord=[]
        for x in range(0,len(SnakeCoord),2):    #risistemo le coordinate iniziali dello snake in modo che vengano prese in pasto da active_contour
            coord=coord+[[SnakeCoord[x],SnakeCoord[x+1]]]
        coord=np.array(coord)
        
        contour=active_contour(Open, coord, alpha=0.8, beta=0.7, w_line= -0.7, w_edge=0.5,  #ottengo lo snake contour che non riesco a disegnare sopra
                                gamma=0.02,
                                bc=None,
                                max_px_move=1.0,
                                max_iterations=200)
        Preprocessing.draw_closing_lines(img,coord)
    

    def draw_closing_lines(img,SnakeCoord):     #funzione per disegnare le linee dello snake trovate
        p=0
        x1=[]
        x2=[]
        y1=[]
        y2=[]
        for item in SnakeCoord:
            if p==0:                            #risistemo le coordinate dello snake di modo tale che vengano ordinate a coppie 
                x1.append(item[0])              #e che vengano prese da cv.line in quanto non riesco ad usare la funzione DrawContours
                y1.append(item[1])              #Sono sicuro ci sia un modo più efficiente e funzionante ma non riesco a trovarlo o farlo funzionare
                p= p + 1

            elif p==1:
                x2.append(item[0])
                y2.append(item[1])
                p=0
        if len(x1)> len(x2):
            x2.append(x1[-1])
            y2.append(y1[-1])
        elif len(x2)>len(x1):
            x1.append(x2[-1])
            y1.append(y2[-1])

        for i in np.arange(len(x1)):
            cv2.line(img, (x1[i],y1[i]),(x2[i],y2[i]),(0,255,0),3)

        cv2.imshow('Img Contours',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Lap(img, hsv):
        grey= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx64= cv2.Sobel(grey, cv2.CV_64F,1,0,ksize=5)
        sobelx=np.uint8(np.absolute(sobelx64))
        sobely64 =cv2.Sobel(grey,cv2.CV_64F, 0,1, ksize=5)
        sobely=np.uint8(np.absolute(sobely64))
        fusion= np.sqrt(sobelx**2+sobely**2)
        cv2.imshow('img',img)
        cv2.imshow('grey',sobelx)
        cv2.imshow('Laplacian', sobely)
        #cv2.imshow('Fusion',fusion)

#                contour_binary = np.zeros(img.shape[:2],
 #                                       dtype=np.uint8)
  #              cv2.drawContours(contour_binary, mask_contours,
   #                                 max_area_pos,
    #                                 255,
     #                                2)
      #          cv2.imshow('bitwise',contour_binary)
       #         cv2.waitKey(0)
        #        cv2.destroyAllWindows()
         #       contour_area = cv2.contourArea(contour)
          #      segmented_img = cv2.bitwise_and(
           #             img, img,
            #            mask=contour_mask)
             #   segmented_img[segmented_img == 0] = 255
 #           else:
  #              print("No contours found")
   #     return







path=os.getcwd()
mystring=IMG_list.List(path)
processing=Preprocessing()
iterations=3
for names in mystring:
    img=cv2.imread(names)
    Open=Preprocessing.PrePro(img)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Preprocessing.Lines(img, Open)
    Preprocessing.Lap(img,Open)

