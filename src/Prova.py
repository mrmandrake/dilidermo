import os, cv2
import numpy, numpy.ma
import matplotlib
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
                    return(str(perc))
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
                    cv2.destroyAllWindows()
                return(immagine)
                    
                    

class Preprocessing:
    def Histograms(immagine):
        channels=[0]
        hist=cv2.calcHist([immagine], channels, mask=None, histSize=[256], ranges=[0,256])
        plt.plot(hist) 
        plt.show()
        return (hist)


mypath=os.getcwd()
myfold=Folder(mypath)
prepro=Preprocessing()
lista_cartelle=os.listdir(mypath)
percorso=myfold.Cartella(mypath, lista_cartelle[4])
immagine=myfold.Apertura(percorso)
Preprocessing.Histograms(immagine)

