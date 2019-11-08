import os, cv2
import numpy, numpy.ma
import matplotlib
import sys
from matplotlib import pyplot as plt
import random 

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
                    cv2.waitKey(0)  
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
        cv2.imshow('CLAHE output', cl)

        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))
        cv2.imshow('limg', limg)

        #-----Converting image from LAB Color model to RGB model--------------------
        finalContrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.imshow('final', finalContrast)
        Preprocessing.Grads(finalContrast,immagine)
        #Preprocessing.RimPix(finalContrast,immagine)
    
    def Grads(finalContrast, immagine):
        kernel=(1,15)
        imm=immagine
        grads=cv2.morphologyEx(finalContrast, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('Grads',grads)
        for Xcoord in range(0,grads.shape[0]):
            for Ycoord in range(0,grads.shape[1]):
                pixs=grads[Xcoord,Ycoord]

                if pixs[0]>=50 & pixs[2]>=50:
                    if pixs[1] >=25:
                        imm[Xcoord,Ycoord]=[255,255,0]
                        print("Trovato pixel")
                        print(imm [Xcoord,Ycoord])
        cv2.imshow("Primo Tentativo", imm)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(immagine,(kernel_size, kernel_size),0)

        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = numpy.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = numpy.copy(immagine) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, numpy.array([]),
                    min_line_length, max_line_gap)
        print(lines)
        points = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        lines_edges = cv2.addWeighted(immagine, 0.8, line_image, 1, 0)
        cv2.imshow('Linee', lines_edges)
                                

    def RimPix (imm):
        BW= cv2.cvtColor(imm, cv2.COLOR_RGB2GRAY)
        ret,BIN=cv2.threshold(BW,35,255,cv2.THRESH_BINARY)
        cv2.imshow('BIN',BIN)
        edged=cv2.Canny(BIN, 20, 30, apertureSize=5)
        lines = cv2.HoughLines(edged,10,numpy.pi/180, 255,255,0)
        for x1 in lines:
                #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
                if flg==1:
                    pts = numpy.array([[x, y]], numpy.int32)
                    flg=flg+1
                else:
                    pts2=numpy.array([[x,y]], numpy.int32)
                    pts=numpy.concatenate((pts, pts2))
                    cv2.polylines(immagine, [pts], True, (0,255,0))
                    flg=1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(immagine,"Tracks Detected", (500, 250), font, 0.5, 255)
        cv2.imshow("Trolley_Problem_Result", immagine)
        cv2.imshow('edge', immagine)
        

            
                
        #Preprocessing.Gradienti(BIN)
    
    #def Gradienti( immagine):
        


  


mypath=os.getcwd()
myfold=Folder(mypath)
prepro=Preprocessing()
lista_cartelle=os.listdir(mypath)
percorso=myfold.Cartella(mypath, lista_cartelle[4])
immagine=myfold.Apertura(percorso)
finalContrast=Preprocessing.Contrasto(immagine)
Preprocessing.Histograms(finalContrast)

