import os, cv2
import numpy, numpy.ma
import matplotlib
import sys
from matplotlib import pyplot as plt
import random
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

#Definisco la classe di ricerca della cartella delle immagini
class Folder:
    def __init__(self, mypath):
        self.mypath = mypath

        
    def Cartella(self, mypath, elementi):
 # Elenco di cartelle presenti nella CDirectory
        for pths,sbdir,subfil in os.walk(elementi):
            for names in sbdir:
                if names.endswith("2017"):
                    perc=os.path.join(mypath,str(pths),str(names))
                    os.chdir(str(perc))
                    return(perc)
                    break
                break
                
        
    
    def Apertura(self,perc):
        for phts, sbdir, sbfil in os.walk(perc):
            for photos in sbfil:
                if photos.endswith('bmp'):
                    perc_img=os.path.join(phts,str(photos))
                    immagine=cv2.imread(perc_img)
                    cv2.imshow('immagine',immagine)
                    #cv2.waitKey(0)  
                    #cv2.destroyAllWindows()
                    return(immagine)
                    
                    

class Preprocessing:

    def Histograms(immagine):
        channels=[0]
        hist=cv2.calcHist([immagine], channels, mask=None, histSize=[256], ranges=[0,256])
        plt.plot(hist) 
        plt.show()
        return (hist)
        
    def Contrasto (immagine):
        img=immagine
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #cv2.imshow("lab",lab)

        #-----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)
        #cv2.imshow('l_channel', l)
        #cv2.imshow('a_channel', a)
        #cv2.imshow('b_channel', b)

        #-----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4,4))
        cl = clahe.apply(l)

        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))

        #-----Converting image from LAB Color model to RGB model--------------------
        finalContrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        Preprocessing.Grads(finalContrast,immagine)
        #Preprocessing.RimPix(finalContrast,immagine)
    
    def Grads(finalContrast, immagine):
        kernel=(1,15)
        imm=immagine
        grads=cv2.morphologyEx(finalContrast, cv2.MORPH_GRADIENT, kernel)
        for Xcoord in range(0,grads.shape[0]):
            for Ycoord in range(0,grads.shape[1]):
                pixs=grads[Xcoord,Ycoord]

                if pixs[0]>=50 & pixs[2]>=50:
                    if pixs[1] >=25:
                        imm[Xcoord,Ycoord]=[255,255,0]
        Preprocessing.HairsPix(imm)
        #Le Coordinate dei pixel non servono, vengono identificate su RimPix,
        #Questa funzione di gradiente resta, ma andrà definita un'altra funzione
        #Per l'identificazione della ROI contenente il Neo.

                                

    def HairsPix (imm):
        #Da implementare: ho trovato le linee dei peli, ora devo creare un kernel, di grandezza variabile, che passi sui peli identificati e applichi
        #una sostituzione delle intensità del valore dei peli con l'intensità gaussiana media delle intensità del neo meno i pixel dei peli. Se invece sono fuori dalla ROI,
        #una loro sostituzione con un valore altamente identificabile (0,255,0).
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(imm,(kernel_size, kernel_size),0)

        low_threshold = 25
        high_threshold = 90
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        cv2.imshow('canny', edges)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = numpy.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 30  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = numpy.copy(immagine) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, numpy.array([]),
                    min_line_length, max_line_gap)
        points = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        lines_edges = cv2.addWeighted(immagine, 0.8, line_image, 1, 0)
        cv2.imshow('Linee', lines_edges)
        Preprocessing.NeoContours(imm, lines_edges)
    
    def NeoContours(immagine, lines_edges):
        # Convert to grayscale
        gray = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 100, 125, cv2.THRESH_BINARY)

        # Downsize image (by factor 4) to speed up morphological operations
        #gray = cv2.resize(gray, dsize=(0, 0), fx=0.25, fy=0.25)

        # Morphological Closing: Get rid of the hole
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))

        # Morphological opening: Get rid of the stuff at the top of the circle
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40)))

        # Resize image to original size
        gray = cv2.resize(gray, dsize=(immagine.shape[1], immagine.shape[0]))

        # Find contours (only most external)
        cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

        # Draw found contour(s) in input image
        image = cv2.drawContours(immagine, cnts, -1, (0, 0, 255), 2)

        cv2.imshow('Contours', image)

        



  


mypath=os.getcwd()
myfold=Folder(mypath)
prepro=Preprocessing()
lista_cartelle=os.listdir(mypath)
percorso=myfold.Cartella(mypath, lista_cartelle[4])
immagine=myfold.Apertura(percorso)
finalContrast=Preprocessing.Contrasto(immagine)
Preprocessing.Histograms(finalContrast)

