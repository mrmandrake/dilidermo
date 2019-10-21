         ###################################################
        #Apertura di immagini multiple
        ###################################################

        #Importo moduli: OS, OpenCV, Numpy

import os, cv2, numpy

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
             
for name in newfolder:
 #i print sono di controllo funzionamento poi spariranno
    path=os.path.join(os.getcwd(), str(name))#controllo su ogni nome di folder, unisco alla cwd
    files=os.listdir(os.path.join(mypath, name))#vedo i file contenuti dentro ogni directory
    for fi in files:
            while fi.endswith('jpg'):
                print(fi)
                percorso=path
                break
                break
                break
print(percorso)

####################################################
            
