# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 22:28:38 2023

@author: Manish Kumar
"""
import cv2
import numpy as np
import statistics
from numpy import copy
from skimage.util import random_noise
img = cv2.imread("C:\\Users\\HARSH NARAYAN PANDEY\Downloads\\Apple Cinnamon Oatmeal Cookies - Cooking Classy.jpeg",1)
#cv2.imshow("Colour Image ",img)
img = cv2.resize(img,(512,512))
cv2.imshow("original",img)
print(img)
print("shape==",img.shape)
print("no. of pixel==",img.size)

#print(img[2,2,0])
p=512
no_noise=0
for i in range(p):
    for j in range(p):
        for k in range(3):
            if(img[i,j,k]==0 or img[i,j,k]==255):
                no_noise+=1
                #print(img[i,j,k])
                
                
print("First no of noise :",no_noise)           

noise_img = random_noise(img, mode='s&p',amount=0.1)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img = np.array(255*noise_img, dtype = 'uint8')

# Display the noise image
noise_img = cv2.resize(noise_img,(512,512))
cv2.imshow('Corrupted Image',noise_img)
#print(noise_img)

cv2.waitKey()
cv2.destroyAllWindows()