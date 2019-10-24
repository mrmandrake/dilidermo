         ###################################################
        #Apertura di immagini multiple
        ###################################################

        #Importo moduli: OS, OpenCV, Numpy

import os, cv2
import numpy, numpy.ma
import matplotlib
from matplotlib import pyplot as plt

############################################################
                #Calcolo istrogramma 
############################################################
def Histogram (grey, mask=None):
        channel=[0]
        hist=cv2.calcHist([grey], channels=[0], mask=None, histSize=[256], ranges=[0,256])
        plt.plot(hist) 
        plt.show()
        return (hist)
############################################################
        
############################################################
#Funzione di preprocessing immagini (Calcolo gradienti)
############################################################
def Gradienti (img, grey):
        laplacian64=cv2.Laplacian(img, cv2.CV_64F)
        laplacian=numpy.unit8(numpy.absolute(laplacian64))
        sobelx64 = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=5)
        sbobelx=numpy.unit8(numpy.absolute(sobelx64))
        sobely64 =cv2.Sobel(img,cv2.CV_64F, 0,1, ksize=5)
        sobely=numpy.unit8(numpy.absolute(sobely64))
        

        matplotlib.pyplot.subplot(2,2,1)
        matplotlib.pyplot.imshow(grey, cmap='gray')
        matplotlib.pyplot.title('Original')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.subplot(2,2,2)
        matplotlib.pyplot.imshow(laplacian,cmap='gray')
        matplotlib.pyplot.title('Laplacian')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.subplot(2,2,3)
        matplotlib.pyplot.imshow(sobelx,cmap='gray')
        matplotlib.pyplot.title('Sobel X')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.subplot(2,2,4)
        matplotlib.pyplot.imshow(sobely,cmap='gray')
        matplotlib.pyplot.title('Sobel Y')
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.show()
        #Controllo directory di lavoro. 
mypath=os.getcwd()  #Attualmente sto lavorando sul Desktop
newfolder=[]
for nome in os.listdir(mypath): #Elenco di cartelle presenti nella CDirectory
        if os.path.isdir(os.path.join(mypath, nome)):#Controllo sulle sole cartelle
         if nome !='\n':
             newfolder=newfolder+list({nome})
             print(newfolder) #Stampo nome a video        

####################################################
#Ingresso nella directory designata
####################################################
imgs=''             
for name in newfolder:
 #i print sono di controllo funzionamento poi spariranno
    path=os.path.join(os.getcwd(), str(name))#controllo su ogni nome di folder, unisco alla cwd
    files=os.listdir(os.path.join(mypath, name))#vedo i file contenuti dentro ogni directory
    for fi in files:
            while fi.endswith('jpg'):
                imgs=imgs+fi
                imgs=imgs+'\n'
                print(fi)
                percorso=path
                break
                break
                break
print(percorso)
os.chdir(percorso)
####################################################
#Apertura sequenziale immagini
####################################################
img_list = imgs.strip('][').split('\n')
for k in img_list:
        if k!='':      #controllo eccessivo, non serve tanto non andranno aperte
                img_path=os.path.join(os.getcwd(),str(k))
                img=cv2.imread(str(img_path))
                grey=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                cv2.imshow("Maledetta immagine",grey)
                Histogram(grey)
                Gradienti (img, grey)
                cv2.waitKey(500)  
                cv2.destroyAllWindows()

######################################################
#Preprocessing
######################################################









