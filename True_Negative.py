# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:51:50 2023

@author: Manish Kumar
"""

import cv2
import random
from numpy import copy
img = cv2.imread("C:\\Users\\HARSH NARAYAN PANDEY\\Desktop\\Project\\project_3_lena.jpg",0)
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
        else:
            continue

print("First no. of noise = ",no_noisy_1)


def add_noise(A):
    row , col = A.shape
    #print(row,col)
    
    val=A.size
    val=int(val*0.8)
    # print("val",val)
    val=int(val/2)
    #number_of_pixels=random.randint(a, b)
    for i in range(val):
       
        y=random.randint(0, row - 1)
        
        x=random.randint(0, col - 1)
         
        A[y][x] = 255
         
    for i in range(val):
       
        y=random.randint(0, row - 1)
         
        x=random.randint(0, col - 1)
         
        A[y][x] = 0
         
    return A



img_1=add_noise(A)
cv2.imshow("10% noise image",img_1)
print(img_1)
p=512
A=copy(img_1)
B = copy(A)
no_noisy_2=0
for i in range(p):
    for j in range(p):
        if(img_1[i][j]==255 or img_1[i][j]==0):
            no_noisy_2=no_noisy_2+1
        

print("Second no. of noisy pixel = ",no_noisy_2)

mid_noisy=0
high_noisy=0
p_max,p_min=0,0
for i in range(p-4):
    for j in range(p-4):
        #th=int(A[i+2,j+2])
        th=0
        th=int(A[i+2,j+3])
        for a in range(5):
            for b in range(5):
                t=A[i+a,j+b]
                p_max=max(p_max,t)
                p_min=min(p_min,t)
                #th+=int(A[i+a,j+b])
        
        #th=th/25
        if(p_min>th or th>p_max):
            A[i+2,j+2]=255
            high_noisy+=1
            continue
        
        d1,d2,d3,d4=0,0,0,0
        ws,wm,wl=0.5,0.25,1
        row,col=i+0,j+0
        no_noisy=0
        d1=d1+int(abs(int(A[row+1,col])-th)*wl)
        d1=d1+int(abs(int(A[row,col])-th)*wm)
        d1+=int(abs(int(A[row+1,col+1])-th)*ws)
        d1=d1+int(abs(int(A[row+3,col+3])-th)*ws)
        d1=d1+int(abs(int(A[row+4,col+4])-th)*wm)
        d1=d1+int(abs(int(A[row+3,col+4])-th)*wl)
        #print(d1)
        
        row,col=i+0,j+2
        d2+=int(abs(int(A[row,col-1])-th)*wl)
        d2+=int(abs(int(A[row,col])-th)*wm)
        d2+=int(abs(int(A[row+1,col])-th)*ws)
        d2+=int(abs(int(A[row+3,col])-th)*ws)
        d2+=int(abs(int(A[row+4,col])-th)*wm)
        d2+=int(abs(int(A[row+4,col+1])-th)*wl)
        
        row,col=i+0,j+4
        d3+=int(abs(int(A[row,col-1])-th)*wl)
        d3+=int(abs(int(A[row,col])-th)*wm)
        d3+=int(abs(int(A[row+1,col-1])-th)*ws)
        d3+=int(abs(int(A[row+3,col-3])-th)*ws)
        d3+=int(abs(int(A[row+4,col-4])-th)*wm)
        d3+=int(abs(int(A[row+4,col-3])-th)*wl)
        
        row,col=i+2,j+0
        d4+=int(abs(int(A[row+1,col])-th)*wl)
        d4+=int(abs(int(A[row,col])-th)*wm)
        d4+=int(abs(int(A[row,col+1])-th)*ws)
        d4+=int(abs(int(A[row,col+3])-th)*ws)
        d4+=int(abs(int(A[row,col+4])-th)*wm)
        d4+=int(abs(int(A[row-1,col+4])-th)*wl)
        
        D=min(d1,d2)
        D=min(D,d3)
        D=min(D,d4)
        
        if(D>th):
            mid_noisy+=1
            B[i+2,j+2]=255


Tp,Tn,Fp,Fn=0,0,0,0
P=512
s_ment_1=0
s_ment_2=0
for i in range(P):
    for j in range(P):
        if((img_1[i,j]==255 or img_1[i,j]==0) and (B[i,j]==255)):
            Tp+=1
        elif((img_1[i,j]==255 or img_1[i,j]==0) and (B[i,j]!=255)):
            Fn+=1
        elif((img_1[i,j]!=255 and img_1[i,j]!=0) and (B[i,j]==255)):
            Fp+=1
        else:
            Tn+=1
            


print("No. of high noisy = ",high_noisy)
print("No. of midium noisy = ",mid_noisy)

print("True Positive = ",Tp)
print("True Negative = ",Tn)
print("False Positive = ",Fp)
print("Fasle Negative = ",Fn)
            
        

                

cv2.imshow("Swastic image ",B)
print(B)

#Total=img.size
Total_Acc=(Tp+Tn)/(Tp+Tn+Fp+Fn)
Error_rate=1-Total_Acc
Precision=Tp/(Fp+Tp)
Re_Cell=Tp/(Tp+Fn)
F1_Score=2*(Precision*Re_Cell)/(Precision+Re_Cell)

print("Total Accoricy = ",Total_Acc)
print("Total Error Rate = ",Error_rate)
print("Total Precision Value =",Precision)
print("Total Re-cell Value = ",Re_Cell)
print("Total F1 Score = ",F1_Score)


cv2.waitKey()
cv2.destroyAllWindows()