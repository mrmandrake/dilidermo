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
                    self.Apertura(perc)
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
                    
                    

class Preprocessing:
    def Histograms(self,immagine):
        channel=[0]
        hist=cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
        plt.plot(hist) 
        plt.show()
        return (hist)


mypath=os.getcwd()
myfold=Folder(mypath)
prepro=Preprocessing()
lista_cartelle=os.listdir(mypath)
for elementi in lista_cartelle:
    myfold.Cartella(mypath, elementi)



