import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

######################### Convertit l'image en niveaux de gris Y
image = cv.imread("C:/Users/asus/Desktop/M1 isii/TIN/Casia v1/017_1_1.bmp")
b,v,r = cv.split(image)         # récupère 3 matrices d'octets séparées
y = 0.299*r + 0.587*v + 0.114*b # opération matricielle
y = y.astype(np.uint8)          # convertit les réels en octets
cv.imshow("image avant l'egalisation", y)

##################### Calcule l'histogramme de l'image################################
histo = np.zeros(256, int)      # prépare un vecteur de 256 zéros
for i in range(0,image.shape[0]):       # énumère les lignes
    for j in range(0,image.shape[1]):   # énumère les colonnes
        histo[y[i,j]] = histo[y[i,j]] + 1

plt.plot(histo)
plt.show()

############################### Calcule l'histogramme cumulé histcum#########################
histcum = np.zeros(256, int)         # prépare un vecteur de 256 zéros
histcum[0] = histo[0]
for i in range(1,256):
    histcum[i] = histo[i] + histcum[i-1]

# Normalise l'histogramme cumulé
nbpixels = y.size
histcum = histcum / nbpixels * 255

plt.plot(histcum)
plt.show()

# Utilise histo cumulé comme table de conversion des niveaux de gris
for i in range(0,y.shape[0]):       # énumère les lignes
    for j in range(0,y.shape[1]):   # énumère les colonnes
        y[i,j] = histcum[y[i,j]]
cv.imshow("image apres egalisation", y)
