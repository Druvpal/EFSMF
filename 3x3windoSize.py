# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:56:16 2023

@author: Manish Kumar
"""

import cv2
#import random
from numpy import copy
# import statistics Library
import math
import statistics 
img = cv2.imread("C:\\Users\\HARSH NARAYAN PANDEY\\Desktop\\Project\\project_3_lena.jpg",0)
img = cv2.resize(img,(5,5))
cv2.imshow("original",img)
print(img)
print("shape==",img.shape)
print("no. of pixel==",img.size)

img[2,2]=0
A=copy(img)
B = copy(A)
p=5
for i in range(p-4):
    for j in range(p-4):
        th=0
        th=int(A[i+2,j+2])
        
        d1,d2,d3,d4=0,0,0,0
        ws,wm,wl=0.25,0.5,1
        
        row,col=i+0,j+0
        stdd_1=int(statistics.stdev([int(A[row+1,col]),int(A[row,col]),int(A[row+1,col+1]),int(A[row+3,col+3])
        ,int(A[row+4,col+4]), int(A[row+3,col+4])]))
        print("stdd_1",stdd_1)
        
        d1=d1+int(abs(int(A[row+1,col])-th)*wl)
        d1=d1+int(abs(int(A[row,col])-th)*wm)
        d1+=int(abs(int(A[row+1,col+1])-th)*ws) 
        d1=d1+int(abs(int(A[row+3,col+3])-th)*ws)
        d1=d1+int(abs(int(A[row+4,col+4])-th)*wm)
        d1=d1+int(abs(int(A[row+3,col+4])-th)*wl)
        
        
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
        print(l)
        l.sort()
        rij=l[0]
        print(l)
        
        l1=[stdd_1,stdd_2,stdd_3,stdd_4]
        print("Standard Derivation",l1)
        l1.sort()
        D1=l1[0]
        D2=l1[1]
        print("Standard Derivation",l1)
        
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
            print(s1)
            R=int(statistics.median(s1))
            print(R)
              
        elif(D1==stdd_4 and D2==stdd_3):
            row,col=i+2,j+0
            s1=[int(A[row+1,col]),int(A[row,col]),int(A[row,col+1]),int(A[row,col+3])
            ,int(A[row,col+4]),int(A[row-1,col+4]),int(A[row,col+1]),int(A[row,col+3])
            ,int(A[row,col+1]),int(A[row,col+3]),int(A[row-1,col+3]),int(A[row+1,col+1])]
            R=int(statistics.median(s1))
            
            
            
       # print(R)   
        #Fuzzy switching alorithm
        T1=125
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
        B[i+2,j+2]=th+miq*(R-th)
           



  
from math import log10, sqrt
import cv2
import numpy as np
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def main():
     original = cv2.imread("img")
     compressed = cv2.imread("C", 1)
     value = PSNR(original, compressed)
     print(f"PSNR value is {value} dB")
       
if __name__ == "__main__":
    main()
#mse=0
#t=0
#for i in range(5):
 #   for j in range(5):
  #      t=abs(B[i,j]-A[i,j])
   #     mse+=pow(t,2)


#mse=mse/25
#print("MSE",mse)
#mse=(255*255)/mse

#psnr=10*(math.log10(mse))
#print("PSNR = ",psnr)
#cv2.imshow("original",img)
print(img)
cv2.imshow("B",B)
print(A)
cv2.waitKey()
cv2.destroyAllWindows()