import os
import sys
from interface import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image



class MyMainWindow(QDialog):
    imgName = ''
    def __init__(self):

        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.file_open)
        self.ui.pushButton_4.clicked.connect(self.checkPath)
        self.ui.pushButton_2.clicked.connect(self.histogramme)
        self.ui.pushButton_3.clicked.connect(self.lissage)
        self.ui.pushButton_6.clicked.connect(self.segmentation)
        self.ui.pushButton_8.clicked.connect(self.operationMorphologique)
        #print(self.checkPath())
    def checkPath(self) :
        image_path = self.ui.lineEdit.text()
        if os.path.isfile(image_path):
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QPixmap(image_path)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.graphicsView_3.setScene(scene)
    def file_open(self):
             name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
             print(str(name)[2:-19])
             self.ui.lineEdit.setText(str(name)[2:-19])
##########################################################################################################################
######################################erosion######################################################################

    def histogramme(self):
        a = str(self.ui.lineEdit.text())
        image = cv.imread(a)
        #image = cv.imread("C:/Users/asus/Desktop/M1 isii/TIN/Casia v1/017_1_1.bmp")
        b, v, r = cv.split(image)  # récupère 3 matrices d'octets séparées
        y = 0.299 * r + 0.587 * v + 0.114 * b  # opération matricielle
        y = y.astype(np.uint8)  # convertit les réels en octets
         #cv.imshow("image avant l'egalisation", y)

        ##################### Calcule l'histogramme de l'image################################
        histo = np.zeros(256, int)  # prépare un vecteur de 256 zéros
        for i in range(0, image.shape[0]):  # énumère les lignes
            for j in range(0, image.shape[1]):  # énumère les colonnes
                histo[y[i, j]] = histo[y[i, j]] + 1

        plt.plot(histo)
        #plt.show()

        ############################### Calcule l'histogramme cumulé histcum#########################
        histcum = np.zeros(256, int)  # prépare un vecteur de 256 zéros
        histcum[0] = histo[0]
        for i in range(1, 256):
            histcum[i] = histo[i] + histcum[i - 1]

        # Normalise l'histogramme cumulé
        nbpixels = y.size
        histcum = histcum / nbpixels * 255

        plt.plot(histcum)
        #plt.show()
        # Utilise histo cumulé comme table de conversion des niveaux de gris
        for i in range(0, y.shape[0]):  # énumère les lignes
            for j in range(0, y.shape[1]):  # énumère les colonnes
                y[i, j] = histcum[y[i, j]]
        im = Image.fromarray(y)
        globals() ['imgName'] = 'your_file.jpg'
        global path
        path = str(self.ui.lineEdit.text())[2:-25]

        im.save("your_file.jpg")
        #cv.imshow("image apres egalisation", y)
        #cv.waitKey(0)
        #if os.path.isfile(a):
        scene = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap("your_file.jpg")
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.graphicsView_2.setScene(scene)
#############################################################################################################
    ################################Lissage##################################################################
    def lissage(self):

        # Read the image
        img_noisy1 = cv.imread(globals() ['imgName'], 0)
        # cv2.imshow('image avant le lissage(bruite)',img_noisy1)
        # Obtain the number of rows and columns
        # of the image
        m, n = img_noisy1.shape

        # Traverse the image. For every 3X3 area,
        # find the median of the pixels and
        # replace the ceter pixel by the median
        img_new1 = np.zeros([m, n])

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = [img_noisy1[i - 1, j - 1],
                        img_noisy1[i - 1, j],
                        img_noisy1[i - 1, j + 1],
                        img_noisy1[i, j - 1],
                        img_noisy1[i, j],
                        img_noisy1[i, j + 1],
                        img_noisy1[i + 1, j - 1],
                        img_noisy1[i + 1, j],
                        img_noisy1[i + 1, j + 1]]

                temp = sorted(temp)
                img_new1[i, j] = temp[4]

        img_new1 = img_new1.astype(np.uint8)
        cv.imwrite('mediane_image.png', img_new1)
        #cv2.imshow('image apres le lissage', img_new1)
        #cv2.waitKey(0)
        globals() ['imgName']='mediane_image.png'
        print("fin de lissage")
        scene = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap('mediane_image.png')
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.graphicsView_2.setScene(scene)
        ################################################################################################################"
        ##################################################################################################################

    def converged(self, centroids, old_centroids):
            if len(old_centroids) == 0:
                return False
            if len(centroids) <= 5:
                a = 1
            elif len(centroids) <= 10:
                a = 2
            else:
                a = 4
            for i in range(0, len(centroids)):
                cent = centroids[i]
                old_cent = old_centroids[i]

                if ((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and (
                        (int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a)) and (
                        (int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a)):
                    continue
                else:
                    return False
            return True





    def getMin(self,pixel, centroids):
        minDist = 99999
        minIndex = 0
        for i in range(0, len(centroids)):
            d = np.sqrt(int((centroids[i][0] - pixel[0])) ** 2 + int((centroids[i][1] - pixel[1])) ** 2 + int(
                (centroids[i][2] - pixel[2])) ** 2)
            if d < minDist:
                minDist = d
                minIndex = i
        return minIndex







    def assignPixels(self,centroids):
        im = Image.open(globals() ['imgName'])
        im = im.convert('RGB')
        img_width, img_height = im.size
        px = im.load()
        clusters = {}
        for x in range(0, img_width):
            for y in range(0, img_height):
                p = px[x, y]
                minIndex = self.getMin(px[x, y], centroids)
                try:
                    clusters[minIndex].append(p)
                except KeyError:
                    clusters[minIndex] = [p]
        return clusters



    def adjustCentroids(self,centroids, clusters):
        new_centroids = []
        keys = sorted(clusters.keys())
        # print(keys)
        for k in keys:
            n = np.mean(clusters[k], axis=0)
            new = (int(n[0]), int(n[1]), int(n[2]))
            new_centroids.append(new)
        return new_centroids






    def startKmeans(self,someK):
        im = Image.open(globals() ['imgName'])
        im = im.convert('RGB')
        img_width, img_height = im.size
        px = im.load()

        centroids = []
        old_centroids = []
        # rgb_range = ImageStat.Stat(im).extrema
        i = 1
        # Initializes someK number of centroids for the clustering
        for k in range(0, someK):

            cent = px[np.random.randint(0, img_width), np.random.randint(0, img_height)]
            centroids.append(cent)
        while not self.converged(centroids, old_centroids) and i <= 20:
            i += 1
            old_centroids = centroids  # Make the current centroids into the old centroids
            clusters = self.assignPixels(centroids)  # Assign each pixel in the image to their respective centroids
            centroids = self.adjustCentroids(old_centroids,
                                        clusters)  # Adjust the centroids to the center of their assigned pixels
        return centroids



    def drawWindow(self,result):
        im = Image.open( globals() ['imgName'])
        im = im.convert('RGB')
        img_width, img_height = im.size
        px = im.load()
        img = Image.new('RGB', (img_width, img_height), "white")
        p = img.load()
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                RGB_value = result[self.getMin(px[x, y], result)]
                p[x, y] = RGB_value

        return img




    def segmentation(self):
        k_input = 4
        result = self.startKmeans(k_input)
        img = self.drawWindow(result)
        img.save('segmentation_image.png')
        globals() ['imgName'] ='segmentation_image.png'
        print("fin de segmentation")
        scene = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap('segmentation_image.png')
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.graphicsView_2.setScene(scene)

###########################################################################################################################
###########################################################################################################################
    def converte_to_black_white(self,image):
        # getting the threshold value
        thresholdValue = np.mean(image)

        # getting the dimensions of the image
        xDim, yDim = image.shape

        # turn the image into a black and white image
        for i in range(xDim):
            for j in range(yDim):
                if (image[i][j] > thresholdValue):
                    image[i][j] = 255
                else:
                    image[i][j] = 0
        return image

    def erosion(self,image, se, K):
        # Acquire size of the image
        m, n = image.shape
        # Show the image
        plt.imshow(image, cmap="gray")
        # Define the structuring element
        # k= 11,15,45 -Different sizes of the structuring element
        # k= se.shape
        constant = (K - 1) // 2
        # Define new image
        imgErode = np.zeros((m, n), dtype=np.uint8)
        # Erosion without using inbuilt cv2 function for morphology
        for i in range(constant, m - constant):
            for j in range(constant, n - constant):
                temp = image[i - constant:i + constant + 1, j - constant:j + constant + 1]
                product = temp * se
                imgErode[i, j] = np.min(product)
        return imgErode

    def Delatation(self,image, se, constant1):
        # Acquire size of the image
        p, q = image.shape
        # Show the image
        plt.imshow(image, cmap="gray")
        # Define new image to store the pixels of dilated image
        imgDilate = np.zeros((p, q), dtype=np.uint8)
        # Define the structuring element

        # Dilation operation without using inbuilt CV2 function
        for i in range(constant1, p - constant1):
            for j in range(constant1, q - constant1):
                temp = image[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]
                product = temp * se
                imgDilate[i, j] = np.max(product)
        return imgDilate

    def operationMorphologique(self):
        img = cv.imread("segmentation_image.png", 0)
        #img = self.converte_to_black_white(img)
        k = 5
        SE = np.ones((k, k), dtype=np.uint8)
        SED = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
         #AeB = self.erosion(img, SE, k)
        AdB = self.Delatation(img, SED, 1)
        print("fin de delatation")
         #AoB = self.Delatation(AeB, SED, 1)
        AcB = self.erosion(AdB, SE, k)
        print("fin de erosion")
        cv.imwrite("eye.png", AcB)
        print("fin de operation mophologique")
        scene = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap('eye.png')
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.graphicsView_2.setScene(scene)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    class_instance = MyMainWindow()
    class_instance.show()
    sys.exit(app.exec_())
