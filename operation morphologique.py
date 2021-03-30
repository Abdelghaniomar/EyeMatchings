# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:05:19 2021

@author: issom
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 18:56:11 2021

@author: asus
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
#Use of opening and closing for morphological filtering
#Perform the following operation on the noisy fingerprint image
# [(((AoB)d B) e B)]
#AoB= (A e B) d B
#o=opening, e=erosion,d=dilation
#Here inbuilt function of erosion and dilation from cv2 module is used.
#To form the structuring element also, inbuilt function from cv2 is used

def converte_to_black_white(image):
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

######################Function for erosion
def erosion(image,se,K):
#Acquire size of the image
    m,n= image.shape 
#Show the image
    plt.imshow(image, cmap="gray")
# Define the structuring element
# k= 11,15,45 -Different sizes of the structuring element
    #k= se.shape 
    constant=(K-1)//2
#Define new image
    imgErode= np.zeros((m,n), dtype=np.uint8)
#Erosion without using inbuilt cv2 function for morphology
    for i in range(constant, m-constant):
        for j in range(constant,n-constant):
            temp= image[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*se
            imgErode[i,j]= np.min(product)
    return imgErode

######################Function for dilation
def Delatation(image,se,constant1):
    #Acquire size of the image
    p,q= image.shape
    #Show the image
    plt.imshow(image, cmap="gray")
    #Define new image to store the pixels of dilated image
    imgDilate= np.zeros((p,q), dtype=np.uint8)
    #Define the structuring element 
    
    #Dilation operation without using inbuilt CV2 function
    for i in range(constant1, p-constant1):
      for j in range(constant1,q-constant1):
        temp= image[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
        product= temp*se
        imgDilate[i,j]= np.max(product)
    return imgDilate
#Read the image for dilation
img= cv2.imread("segmentation_image.png",0)
img=converte_to_black_white(img)
img_eye=cv2.imwrite("eye.png", img)

#####################Define the structuring element using inbuilt CV2 function
k=5
SE= np.ones((k,k), dtype=np.uint8)
SED= np.array([[1,1,1], [1,1,1],[1,1,1]])
#Erode the image
AeB= erosion(img,SE,k)
#dilate the image
AdB= Delatation(img,SED,1)
#Dilate the eroded image. This gives opening operation 
AoB= Delatation(AeB, SED,1) 

 #erode the dilate image , gives you the close operation
AcB= erosion(AdB, SE,k)


#Plot all the images
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(3,2,2)
plt.title(" erosion : E(A,B)")
plt.imshow(AeB, cmap="gray")
cv2.imshow("erosion",AeB)
cv2.waitKey(0)

plt.subplot(3,2,3)
plt.title("delatation : D(A, B)")
plt.imshow(AoB, cmap="gray")
cv2.imshow("delatation",AeB)
cv2.waitKey(0)

plt.subplot(3,2,4)
plt.title(" open : O(A,B)")
plt.imshow(AoB, cmap="gray")
cv2.imshow("open",AoB)
cv2.waitKey(0)

plt.subplot(3,2,5)
plt.title(" close : C(A,B)")
plt.imshow(AcB, cmap="gray")
cv2.imshow("close",AcB)
cv2.waitKey(0)
#Save the filtered image
cv2.imwrite("eye.png", AcB)