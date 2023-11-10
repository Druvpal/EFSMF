# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:40:50 2023

@author: Manish Kumar
"""

#from math import log10, sqrt 
import cv2
import random
from numpy import copy
import math
# import statistics Library
import statistics 
img = cv2.imread("C:\\Users\\HARSH NARAYAN PANDEY\\Downloads\\sodapdf-converted.jpg",0)
img = cv2.resize(img,(512,512))
cv2.imshow("original",img)
print(img)
print("shape==",img.shape)
print("no. of pixel==",img.size)

no_noisy_1=0
A=img
w_s=5
P=512

for i in range(P):
    for j in range(P):
        if(A[i,j]==255 or A[i,j]==0):
            no_noisy_1+=1
       

print("First no. of noise = ",no_noisy_1)

def add_noise(A):
    row , col = A.shape
    #print(row,col)
    
    val=A.size
    val=int(val*0.1)
    # print("val",val)
    val=int(val/2)
    #number_of_pixels=random.randint(a, b)
    for i in range(val):
       
        y=random.randint(0, row - 1)
        
        x=random.randint(0, col - 1)
         
        if(int(A[y][x])==255 or int(A[y][x])==0):
            
            i=i-1
        else:
            
            A[y][x] = 255
                           
    for i in range(val):
       
        y=random.randint(0, row - 1)
         
        x=random.randint(0, col - 1)
         
        if(int(A[y][x])==255 or int(A[y][x])==0):
            
            i=i-1
        else:
            
            A[y][x] = 0
         
    return A


img_1=add_noise(A)
cv2.imshow("80% noise image",img_1)
print(img_1)
p=512
A=copy(img_1)
B = copy(A)
#cv2.imshow("A noise image",A)
#print(A)
#cv2.imshow("B noise image",B)
#print(B)
no_noisy_2=0
for i in range(p):
    for j in range(p):
        if(img_1[i][j]==255 or img_1[i][j]==0):
            no_noisy_2+=1
        

print("Second no. of noisy pixel = ",no_noisy_2)

mid_noisy=0
high_noisy=0
p_max,p_min=0,0

B=copy(A)
C = copy(B)
for i in range(p-4):
    for j in range(p-4):
        
        th=0 
        th=int(A[i+2,j+2])
       
        d1,d2,d3,d4=0,0,0,0
        ws,wm,wl=0.25,0.5,1
        row,col=i+0,j+0
        no_noisy=0 
        
        stdd_1=int(statistics.stdev([int(A[row+1,col]),int(A[row,col]),int(A[row+1,col+1]),int(A[row+3,col+3])
        ,int(A[row+4,col+4]), int(A[row+3,col+4])]))
        #mean_1=mean_1/6                                                                                                                                          
        d1=d1+int(abs(int(A[row+1,col])-th)*wl)
        d1=d1+int(abs(int(A[row,col])-th)*wm)
        d1+=int(abs(int(A[row+1,col+1])-th)*ws) 
        d1=d1+int(abs(int(A[row+3,col+3])-th)*ws)
        d1=d1+int(abs(int(A[row+4,col+4])-th)*wm)
        d1=d1+int(abs(int(A[row+3,col+4])-th)*wl)
        #print(d1)
        
        row,col=i+0,j+2
        
        stdd_2=int(statistics.stdev([int(A[row,col-1]),int(A[row,col]),int(A[row+1,col]),int(A[row+3,col])
        ,int(A[row+4,col]),int(A[row+4,col+1])]))
        
        d2+=int(abs(int(A[row,col-1])-th)*wl)
        d2+=int(abs(int(A[row,col])-th)*wm)
        d2+=int(abs(int(A[row+1,col])-th)*ws)
        d2+=int(abs(int(A[row+3,col])-th)*ws)
        d2+=int(abs(int(A[row+4,col])-th)*wm)
        d2+=int(abs(int(A[row+4,col+1])-th)*wl)
        
        row,col=i+0,j+4
        
        stdd_3=int(statistics.stdev([int(A[row,col-1]),int(A[row,col]),int(A[row+1,col-1]),int(A[row+3,col-3])
        ,int(A[row+4,col-4]),int(A[row+4,col-3])]))
        
        d3+=int(abs(int(A[row,col-1])-th)*wl)
        d3+=int(abs(int(A[row,col])-th)*wm)
        d3+=int(abs(int(A[row+1,col-1])-th)*ws)
        d3+=int(abs(int(A[row+3,col-3])-th)*ws)
        d3+=int(abs(int(A[row+4,col-4])-th)*wm)
        d3+=int(abs(int(A[row+4,col-3])-th)*wl)
        
        row,col=i+2,j+0
        
        stdd_4=int(statistics.stdev([int(A[row+1,col]),int(A[row,col]),int(A[row,col+1]),int(A[row,col+3])
        ,int(A[row,col+4]),int(A[row-1,col+4])]))
        
        d4+=int(abs(int(A[row+1,col])-th)*wl)
        d4+=int(abs(int(A[row,col])-th)*wm)
        d4+=int(abs(int(A[row,col+1])-th)*ws)
        d4+=int(abs(int(A[row,col+3])-th)*ws)
        d4+=int(abs(int(A[row,col+4])-th)*wm)
        d4+=int(abs(int(A[row-1,col+4])-th)*wl)
        
        l=[d1,d2,d3,d4]
        l.sort()
        rij=l[0]
        

        
        l1=[stdd_1,stdd_2,stdd_3,stdd_4]
        l1.sort()
        D1=l1[0]
        D2=l1[1]
        
        s1=[]
        R=0
        if(D1==stdd_1 and D2==stdd_2):
            row,col=i+0,j+0
            s1=[int(A[row+1,col]),int(A[row,col]),int(A[row+1,col+1]),int(A[row+3,col+3])
            ,int(A[row+4,col+4]), int(A[row+3,col+4]),int(A[row+1,col+1]),int(A[row+3,col+3])
            ,int(A[row+1,col+1]),int(A[row+3,col+3]),int(A[row+1,col+2]),int(A[row+3,col+2])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_1 and D2==stdd_3):
            row,col=i+0,j+0
            s1=[int(A[row+1,col]),int(A[row,col]),int(A[row+1,col+1]),int(A[row+3,col+3])
            ,int(A[row+4,col+4]), int(A[row+3,col+4]), int(A[row+1,col+1]),int(A[row+3,col+3])
            ,int(A[row+1,col+1]),int(A[row+3,col+3]),int(A[row+1,col+3]),int(A[row+3,col+1])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_1 and D2==stdd_4):
            row,col=i+0,j+0
            s1=[int(A[row+1,col]),int(A[row,col]),int(A[row+1,col+1]),int(A[row+3,col+3])
            ,int(A[row+4,col+4]), int(A[row+3,col+4]),int(A[row+1,col+1]),int(A[row+3,col+3])
            ,int(A[row+1,col+1]),int(A[row+3,col+3]),int(A[row+2,col+1]),int(A[row+2,col+3])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_2 and D2==stdd_1):
            row,col=i+0,j+2
            s1=[int(A[row,col-1]),int(A[row,col]),int(A[row+1,col]),int(A[row+3,col])
            ,int(A[row+4,col]),int(A[row+4,col+1]),int(A[row+1,col]),int(A[row+3,col])
            ,int(A[row+1,col]),int(A[row+3,col]),int(A[row+1,col-1]),int(A[row+3,col+1])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_2 and D2==stdd_3):
            row,col=i+0,j+2
            s1=[int(A[row,col-1]),int(A[row,col]),int(A[row+1,col]),int(A[row+3,col])
            ,int(A[row+4,col]),int(A[row+4,col+1]),int(A[row+1,col]),int(A[row+3,col])
            ,int(A[row+1,col]),int(A[row+3,col]),int(A[row+1,col+1]),int(A[row+3,col-1])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_2 and D2==stdd_4):
            row,col=i+0,j+2
            s1=[int(A[row,col-1]),int(A[row,col]),int(A[row+1,col]),int(A[row+3,col])
            ,int(A[row+4,col]),int(A[row+4,col+1]),int(A[row+1,col]),int(A[row+3,col])
            ,int(A[row+1,col]),int(A[row+3,col]),int(A[row+2,col-1]),int(A[row+2,col+1])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_3 and D2==stdd_1):
            row,col=i+0,j+4
            s1=[int(A[row,col-1]),int(A[row,col]),int(A[row+1,col-1]),int(A[row+3,col-3])
            ,int(A[row+4,col-4]),int(A[row+4,col-3]),int(A[row+1,col-1]),int(A[row+3,col-3])
            ,int(A[row+1,col-1]),int(A[row+3,col-3]),int(A[row+1,col-3]),int(A[row+3,col-1])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_3 and D2==stdd_2):
            row,col=i+0,j+4
            s1=[int(A[row,col-1]),int(A[row,col]),int(A[row+1,col-1]),int(A[row+3,col-3])
            ,int(A[row+4,col-4]),int(A[row+4,col-3]),int(A[row+1,col-1]),int(A[row+3,col-3])
            ,int(A[row+1,col-1]),int(A[row+3,col-3]),int(A[row+1,col-2]),int(A[row+3,col-2])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_3 and D2==stdd_4):
            row,col=i+0,j+4
            s1=[int(A[row,col-1]),int(A[row,col]),int(A[row+1,col-1]),int(A[row+3,col-3])
            ,int(A[row+4,col-4]),int(A[row+4,col-3]),int(A[row+1,col-1]),int(A[row+3,col-3])
            ,int(A[row+1,col-1]),int(A[row+3,col-3]),int(A[row+2,col-3]),int(A[row+2,col-1])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_4 and D2==stdd_1):
            row,col=i+2,j+0
            s1=[int(A[row+1,col]),int(A[row,col]),int(A[row,col+1]),int(A[row,col+3])
            ,int(A[row,col+4]),int(A[row-1,col+4]),int(A[row,col+1]),int(A[row,col+3])
            ,int(A[row,col+1]),int(A[row,col+3]),int(A[row-1,col+1]),int(A[row+1,col+3])]
            R=int(statistics.median(s1))
            
        elif(D1==stdd_4 and D2==stdd_2):
            row,col=i+2,j+0
            s1=[int(A[row+1,col]),int(A[row,col]),int(A[row,col+1]),int(A[row,col+3])
            ,int(A[row,col+4]),int(A[row-1,col+4]),int(A[row,col+1]),int(A[row,col+3])
            ,int(A[row,col+1]),int(A[row,col+3]),int(A[row-1,col+2]),int(A[row+1,col+2])]
            R=int(statistics.median(s1))
              
        elif(D1==stdd_4 and D2==stdd_3):
            row,col=i+2,j+0
            s1=[int(A[row+1,col]),int(A[row,col]),int(A[row,col+1]),int(A[row,col+3])
            ,int(A[row,col+4]),int(A[row-1,col+4]),int(A[row,col+1]),int(A[row,col+3])
            ,int(A[row,col+1]),int(A[row,col+3]),int(A[row-1,col+3]),int(A[row+1,col+1])]
            R=int(statistics.median(s1))
            
        
        #Fuzzy switching alorithm
        T1=80
        T2=135
        miq=0
        if(rij<T1):
            miq=0
        elif(rij>=T1 and rij<((T1+T2)/2)):
            miq=2*((rij-T1/T2-T1)*(rij-T1/T2-T1))
        elif(rij>=(T1+T2)/2 and T2>rij):
            miq=1-2*((rij-T1/T2-T1)*(rij-T1/T2-T1))
        elif(rij>=T2):
            miq=1
        
        #restored the pixel
        C[i+2,j+2]=th+miq*(R-th)        
        
        
        
import numpy as np

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

value = psnr(img, C)
print("PSNR",value)


def immse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return mse

MSE1=immse(img,B);
MSE2=immse(img,C);
IEF=MSE1/MSE2;
print('The Image Enhancement Factor is %.2f',IEF);

no_noisy_3=0
for i in range(p):
    for j in range(p):
        if(C[i][j]==255 or C[i][j]==0):
            no_noisy_3+=1
        
cv2.imshow("Filtered Image :",C)
print(C)
print("Final Noise ",no_noisy_3)

cv2.waitKey()
cv2.destroyAllWindows()