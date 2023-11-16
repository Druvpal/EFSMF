# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 19:55:59 2022

@author: Manish Kumar
"""

import cv2
img = cv2.imread("C:\\Users\\HARSH NARAYAN PANDEY\\Desktop\\Project\\3rdpic.png",0)
img = cv2.resize(img,(3,3))
cv2.imshow("original",img)
print(img)
print("shape==",img.shape)
print("no. of pixel==",img.size)

A=img
th=A[1,1]
d1=abs(A[0,0]-th)+abs(A[2,2]-th)
d2=abs(A[1,0]-th)+abs(A[1,2]-th)
d3=abs(A[2,0]-th)+abs(A[0,2]-th)
d4=abs(A[0,1]-th)+abs(A[2,1]-th)

D=min(d1,d2)
D=min(D,d3)
D=min(D,d4)
print("D_min= ",D)
print("Thresold Value",th)

if(D>th):
    print("Noisy image ")
else:
    print("Noisy free image")
    
cv2.waitKey()
cv2.destroyAllWindows()