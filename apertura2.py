         ###################################################
        #Apertura di immagini multiple
        ###################################################

        #Importo moduli: OS, OpenCV, Numpy

import os, cv2, numpy, numpy.ma
from matplotlib import pyplot as plt
############################################################
#Funzione di preprocessing immagini
############################################################
def Preprocess(img):
        ret,img1 = cv2.threshold(img,127,255,cv2.TRESH_BINARY)
        Hist=cv2.calcHist([img1.astype('float32')], channels=[2], mask=(), histSize=[img.size], ranges=[0,img.size])
        cv2.imshow("Istograma",Hist)
        return (Hist)
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
        #if k!='':      controllo eccessivo, non serve tanto non andranno aperte
                img_path=os.path.join(os.getcwd(),str(k))
                img=cv2.imread(str(img_path))
                cv2.imshow("Maledetta immagine",img)
                Preprocess(img)
                cv2.waitKey(0)  
                cv2.destroyAllWindows()

######################################################
#Preprocessing
######################################################

