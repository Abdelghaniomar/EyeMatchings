# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 23:01:48 2021

@author: issom
"""

# import cv2 
# import matplotlib.pyplot as plt


# # reading image
# img1 = cv2.imread('eye.png')  
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# #keypoints
# sift = cv2.xfeatures2d.SIFT_create()
# keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

# img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1)
# plt.imshow(img_1)



# read images
# img1 = cv2.imread('image apres egalisation.png')  
# img2 = cv2.imread('001_1_1.bmp') 

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# figure, ax = plt.subplots(1, 2, figsize=(16, 8))

# ax[0].imshow(img1, cmap='gray')
# ax[1].imshow(img2, cmap='gray')




# import cv2 
# import matplotlib.pyplot as plt


# # read images
# img1 = cv2.imread('eye.png')  
# img2 = cv2.imread('eye2.png') 

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# #sift
# sift = cv2.xfeatures2d.SIFT_create()

# keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
# keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

# len(keypoints_1), len(keypoints_2)





import cv2 
import matplotlib.pyplot as plt


# read images
img1 = cv2.imread('eye.png')
img2 = cv2.imread('eye2.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# #sift
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

# #feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()


# import cv2
# def extractFeatures_SIFT(Imagelist):
#     l = len(Imagelist)
#     featurlist = []
#     for img_path in Imagelist:
#         img = img_path
#         img = cv2.imread(img)
#         sift = cv2.xfeatures2d.SIFT_create()
#         (kps, descriptor) = sift.detectAndCompute(img, None)
#         featurlist += [kps, descriptor]

#     return featurlist










