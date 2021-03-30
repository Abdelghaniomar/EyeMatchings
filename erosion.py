# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 18:04:22 2021

@author: asus
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


#Read the image for erosion
image= cv2.imread("C:/Users/Slash/Desktop/bb.jpg",0)
#def erosion(image,se):
#Acquire size of the image
m,n= image.shape 
#Show the image
plt.imshow(image, cmap="gray")
# Define the structuring element
# k= 11,15,45 -Different sizes of the structuring element
k=11
SE= np.ones((k,k), dtype=np.uint8)
    #k= se.shape 
constant= (k-1)//2
#Define new image
imgErode= np.zeros((m,n), dtype=np.uint8)
#Erosion without using inbuilt cv2 function for morphology
for i in range(constant, m-constant):
    for j in range(constant,n-constant):
        temp= image[i-constant:i+constant+1, j-constant:j+constant+1]
        product= temp*SE
        imgErode[i,j]= np.min(product)
#return imgErode
"""
se= cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) 
plt.figure(figsize=(10,10))
#AeB= erosion(img1,SE)
plt.title(" erosion : E(A,B)")
plt.imshow(AeB, cmap="gray")
plt.subplot(3,2,3)"""
plt.imshow(imgErode,cmap="gray")
cv2.imwrite("Eroded3.png", imgErode)