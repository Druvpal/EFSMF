# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:45:22 2023

@author: Manish Kumar
"""

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
img = cv2.imread("C:\\Users\\HARSH NARAYAN PANDEY\Downloads\\IMG_20201127_094946.jpg",1)
#cv2.imshow("Colour Image ",img)
img = cv2.resize(img,(3,3))
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


# Add salt-and-pepper noise to the image.
noise_img = random_noise(img, mode='gaussian', seed=None, clip=True, amount=0.1)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img = np.array(255*noise_img, dtype = 'uint8')



# Display the noise image
noise_img = cv2.resize(noise_img,(3,3))
cv2.imshow('Corrupted Image',noise_img)
#print(noise_img)
img[4,4,2]=255
print('img',img[4,4,2])

p=512
no_noise1=0
for i in range(p):
    for j in range(p):
        for k in range(3):
            if(noise_img[i,j,k]==0 or noise_img[i,j,k]==255):
                no_noise1+=1
                
                

print("Second no of noise :",no_noise1)
B=copy(noise_img)
C = copy(B)
#cv2.imshow('B',B)
#cv2.imshow('C',C)
print('B',B[4,4,2])
P=512
for i in range(P-4):
    for j in range(P-4):
            tha,thb,thc=0,0,0
            tha=B[i+2,j+2,0]
            thb=B[i+2,j+2,1]
            thc=B[i+2,j+2,2]
            
            a1,a2,a3,a4=0,0,0,0
            b1,b2,b3,b4=0,0,0,0
            c1,c2,c3,c4=0,0,0,0
            
            ws,wm,wl=0.25,0.5,1
            row,col=i+0,j+0
            no_noisy=0 
            
            ssta1=int(statistics.stdev([int(B[row+1,col,0]),int(B[row,col,0]),int(B[row+1,col+1,0]),int(B[row+3,col+3,0])
            ,int(B[row+4,col+4,0]), int(B[row+3,col+4,0])]))
            
            a1+=int(abs(int(B[row+1,col,0])-tha)*wl)
            a1+=int(abs(int(B[row,col,0])-tha)*wm)
            a1+=int(abs(int(B[row+1,col+1,0])-tha)*ws) 
            a1+=int(abs(int(B[row+3,col+3,0])-tha)*ws)
            a1+=int(abs(int(B[row+4,col+4,0])-tha)*wm)
            a1+=int(abs(int(B[row+3,col+4,0])-tha)*wl)
            
            
            sstb1=int(statistics.stdev([int(B[row+1,col,1]),int(B[row,col,1]),int(B[row+1,col+1,1]),int(B[row+3,col+3,1])
            ,int(B[row+4,col+4,1]), int(B[row+3,col+4,1])]))
            
            b1+=int(abs(int(B[row+1,col,1])-thb)*wl)
            b1+=int(abs(int(B[row,col,1])-thb)*wm)
            b1+=int(abs(int(B[row+1,col+1,1])-thb)*ws) 
            b1+=int(abs(int(B[row+3,col+3,1])-thb)*ws)
            b1+=int(abs(int(B[row+4,col+4,1])-thb)*wm)
            b1+=int(abs(int(B[row+3,col+4,1])-thb)*wl)
            
            sstc1=int(statistics.stdev([int(B[row+1,col,2]),int(B[row,col,2]),int(B[row+1,col+1,2]),int(B[row+3,col+3,2])
            ,int(B[row+4,col+4,2]), int(B[row+3,col+4,2])]))
            
            c1+=int(abs(int(B[row+1,col,2])-thc)*wl)
            c1+=int(abs(int(B[row,col,2])-thc)*wm)
            c1+=int(abs(int(B[row+1,col+1,2])-thc)*ws) 
            c1+=int(abs(int(B[row+3,col+3,2])-thc)*ws)
            c1+=int(abs(int(B[row+4,col+4,2])-thc)*wm)
            c1+=int(abs(int(B[row+3,col+4,2])-thc)*wl)
            
            #for second direction 
            
            row,col=i+0,j+2
            
            ssta2=int(statistics.stdev([int(B[row,col-1,0]),int(B[row,col,0]),int(B[row+1,col,0]),int(B[row+3,col,0])
            ,int(B[row+4,col,0]),int(B[row+4,col+1,0])]))
            
            a2+=int(abs(int(B[row,col-1,0])-tha)*wl)
            a2+=int(abs(int(B[row,col,0])-tha)*wm)
            a2+=int(abs(int(B[row+1,col,0])-tha)*ws)
            a2+=int(abs(int(B[row+3,col,0])-tha)*ws)
            a2+=int(abs(int(B[row+4,col,0])-tha)*wm)
            a2+=int(abs(int(B[row+4,col+1,0])-tha)*wl)
            
            
            sstb2=int(statistics.stdev([int(B[row,col-1,1]),int(B[row,col,1]),int(B[row+1,col,1]),int(B[row+3,col,1])
            ,int(B[row+4,col,1]),int(B[row+4,col+1,1])]))
            
            b2+=int(abs(int(B[row,col-1,1])-thb)*wl)
            b2+=int(abs(int(B[row,col,1])-thb)*wm)
            b2+=int(abs(int(B[row+1,col,1])-thb)*ws)
            b2+=int(abs(int(B[row+3,col,1])-thb)*ws)
            b2+=int(abs(int(B[row+4,col,1])-thb)*wm)
            b2+=int(abs(int(B[row+4,col+1,1])-thb)*wl)

            
            
            sstc2=int(statistics.stdev([int(B[row,col-1,2]),int(B[row,col,2]),int(B[row+1,col,2]),int(B[row+3,col,2])
            ,int(B[row+4,col,2]),int(B[row+4,col+1,2])]))
            
            c2+=int(abs(int(B[row,col-1,2])-thc)*wl)
            c2+=int(abs(int(B[row,col,2])-thc)*wm)
            c2+=int(abs(int(B[row+1,col,2])-thc)*ws)
            c2+=int(abs(int(B[row+3,col,2])-thc)*ws)
            c2+=int(abs(int(B[row+4,col,2])-thc)*wm)
            c2+=int(abs(int(B[row+4,col+1,2])-thc)*wl)
            
            #for third direction 
            
                
            row,col=i+0,j+4
            
            ssta3=int(statistics.stdev([int(B[row,col-1,0]),int(B[row,col,0]),int(B[row+1,col-1,0]),int(B[row+3,col-3,0])
            ,int(B[row+4,col-4,0]),int(B[row+4,col-3,0])]))
            
            a3+=int(abs(int(B[row,col-1,0])-tha)*wl)
            a3+=int(abs(int(B[row,col,0])-tha)*wm)
            a3+=int(abs(int(B[row+1,col-1,0])-tha)*ws)
            a3+=int(abs(int(B[row+3,col-3,0])-tha)*ws)
            a3+=int(abs(int(B[row+4,col-4,0])-tha)*wm)
            a3+=int(abs(int(B[row+4,col-3,0])-tha)*wl)
            
            
            sstb3=int(statistics.stdev([int(B[row,col-1,1]),int(B[row,col,1]),int(B[row+1,col-1,1]),int(B[row+3,col-3,1])
            ,int(B[row+4,col-4,1]),int(B[row+4,col-3,1])]))
            
            b3+=int(abs(int(B[row,col-1,1])-thb)*wl)
            b3+=int(abs(int(B[row,col,1])-thb)*wm)
            b3+=int(abs(int(B[row+1,col-1,1])-thb)*ws)
            b3+=int(abs(int(B[row+3,col-3,1])-thb)*ws)
            b3+=int(abs(int(B[row+4,col-4,1])-thb)*wm)
            b3+=int(abs(int(B[row+4,col-3,1])-thb)*wl)
            
            
            sstc3=int(statistics.stdev([int(B[row,col-1,2]),int(B[row,col,2]),int(B[row+1,col-1,2]),int(B[row+3,col-3,2])
            ,int(B[row+4,col-4,2]),int(B[row+4,col-3,2])]))
            
            c3+=int(abs(int(B[row,col-1,2])-thc)*wl)
            c3+=int(abs(int(B[row,col,2])-thc)*wm)
            c3+=int(abs(int(B[row+1,col-1,2])-thc)*ws)
            c3+=int(abs(int(B[row+3,col-3,2])-thc)*ws)
            c3+=int(abs(int(B[row+4,col-4,2])-thc)*wm)
            c3+=int(abs(int(B[row+4,col-3,2])-thc)*wl)
            
            # for forth direction
            
            row,col=i+2,j+0
            
            ssta4=int(statistics.stdev([int(B[row+1,col,0]),int(B[row,col,0]),int(B[row,col+1,0]),int(B[row,col+3,0])
            ,int(B[row,col+4,0]),int(B[row-1,col+4,0])]))
            
            a4+=int(abs(int(B[row+1,col,0])-tha)*wl)
            a4+=int(abs(int(B[row,col,0])-tha)*wm)
            a4+=int(abs(int(B[row,col+1,0])-tha)*ws)
            a4+=int(abs(int(B[row,col+3,0])-tha)*ws)
            a4+=int(abs(int(B[row,col+4,0])-tha)*wm)
            a4+=int(abs(int(B[row-1,col+4,0])-tha)*wl)
            
            
            sstb4=int(statistics.stdev([int(B[row+1,col,1]),int(B[row,col,1]),int(B[row,col+1,1]),int(B[row,col+3,1])
            ,int(B[row,col+4,1]),int(B[row-1,col+4,1])]))
            
            b4+=int(abs(int(B[row+1,col,1])-thb)*wl)
            b4+=int(abs(int(B[row,col,1])-thb)*wm)
            b4+=int(abs(int(B[row,col+1,1])-thb)*ws)
            b4+=int(abs(int(B[row,col+3,1])-thb)*ws)
            b4+=int(abs(int(B[row,col+4,1])-thb)*wm)
            b4+=int(abs(int(B[row-1,col+4,1])-thb)*wl)
            
            sstc4=int(statistics.stdev([int(B[row+1,col,2]),int(B[row,col,2]),int(B[row,col+1,2]),int(B[row,col+3,2])
            ,int(B[row,col+4,2]),int(B[row-1,col+4,2])]))
            
            c4+=int(abs(int(B[row+1,col,2])-thc)*wl)
            c4+=int(abs(int(B[row,col,2])-thc)*wm)
            c4+=int(abs(int(B[row,col+1,2])-thc)*ws)
            c4+=int(abs(int(B[row,col+3,2])-thc)*ws)
            c4+=int(abs(int(B[row,col+4,2])-thc)*wm)
            c4+=int(abs(int(B[row-1,col+4,2])-thc)*wl)
            
            l1=[a1,a2,a3,a4]
            l1.sort()
            aij=l1[0]
            
            l2=[b1,b2,b3,b4]
            l2.sort()
            bij=l2[0]
            
            l3=[c1,c2,c3,c4]
            l3.sort()
            cij=l3[0]
            
            la1=[ssta1,ssta2,ssta3,ssta4]
            la1.sort()
            A1=la1[0]
            A2=la1[1]
            
            lb1=[sstb1,sstb2,sstb3,sstb4]
            lb1.sort()
            B1=lb1[0]
            B2=lb1[1]
            
            lc1=[sstc1,sstc2,sstc3,sstc4]
            lc1.sort()
            C1=lc1[0]
            C2=lc1[1]
            
            s1=[]
            s2=[]
            s3=[]
            R1,R2,R3=0,0,0
            
            if(A1==ssta1 and A2==ssta2):
                row,col=i+0,j+0
                s1=[int(B[row+1,col,0]),int(B[row,col,0]),int(B[row+1,col+1,0]),int(B[row+3,col+3,0])
                ,int(B[row+4,col+4,0]), int(B[row+3,col+4,0]),int(B[row+1,col+1,0]),int(B[row+3,col+3,0])
                ,int(B[row+1,col+1,0]),int(B[row+3,col+3,0]),int(B[row+1,col+2,0]),int(B[row+3,col+2,0])]
                R1=int(statistics.median(s1))
            
               
            elif(A1==ssta1 and A2==ssta3):
                row,col=i+0,j+0
                s1=[int(B[row+1,col,0]),int(B[row,col,0]),int(B[row+1,col+1,0]),int(B[row+3,col+3,0])
                ,int(B[row+4,col+4,0]), int(B[row+3,col+4,0]), int(B[row+1,col+1,0]),int(B[row+3,col+3,0])
                ,int(B[row+1,col+1,0]),int(B[row+3,col+3,0]),int(B[row+1,col+3,0]),int(B[row+3,col+1,0])]
                R1=int(statistics.median(s1))
            
            elif(A1==ssta1 and A2==ssta4):
                row,col=i+0,j+0
                s1=[int(B[row+1,col,0]),int(B[row,col,0]),int(B[row+1,col+1,0]),int(B[row+3,col+3,0])
                ,int(B[row+4,col+4,0]), int(B[row+3,col+4,0]),int(B[row+1,col+1,0]),int(B[row+3,col+3,0])
                ,int(B[row+1,col+1,0]),int(B[row+3,col+3,0]),int(B[row+2,col+1,0]),int(B[row+2,col+3,0])]
                R1=int(statistics.median(s1))
                
            elif(A1==ssta2 and A2==ssta1):
                row,col=i+0,j+2
                s1=[int(B[row,col-1,0]),int(B[row,col,0]),int(B[row+1,col,0]),int(B[row+3,col,0])
                ,int(B[row+4,col,0]),int(B[row+4,col+1,0]),int(B[row+1,col,0]),int(B[row+3,col,0])
                ,int(B[row+1,col,0]),int(B[row+3,col,0]),int(B[row+1,col-1,0]),int(B[row+3,col+1,0])]
                R1=int(statistics.median(s1))
                
            elif(A1==ssta2 and A2==ssta3):
                row,col=i+0,j+2
                s1=[int(B[row,col-1,0]),int(B[row,col,0]),int(B[row+1,col,0]),int(B[row+3,col,0])
                ,int(B[row+4,col,0]),int(B[row+4,col+1,0]),int(B[row+1,col,0]),int(B[row+3,col,0])
                ,int(B[row+1,col,0]),int(B[row+3,col,0]),int(B[row+1,col+1,0]),int(B[row+3,col-1,0])]
                R1=int(statistics.median(s1))
                
            elif(A1==ssta2 and A2==ssta4):
                row,col=i+0,j+2
                s1=[int(B[row,col-1,0]),int(B[row,col,0]),int(B[row+1,col,0]),int(B[row+3,col,0])
                ,int(B[row+4,col,0]),int(B[row+4,col+1,0]),int(B[row+1,col,0]),int(B[row+3,col,0])
                ,int(B[row+1,col,0]),int(B[row+3,col,0]),int(B[row+2,col-1,0]),int(B[row+2,col+1,0])]
                R1=int(statistics.median(s1))
                
            elif(A1==ssta3 and A2==ssta1):
                row,col=i+0,j+4
                s1=[int(B[row,col-1,0]),int(B[row,col,0]),int(B[row+1,col-1,0]),int(B[row+3,col-3,0])
                ,int(B[row+4,col-4,0]),int(B[row+4,col-3,0]),int(B[row+1,col-1,0]),int(B[row+3,col-3,0])
                ,int(B[row+1,col-1,0]),int(B[row+3,col-3,0]),int(B[row+1,col-3,0]),int(B[row+3,col-1,0])]
                R1=int(statistics.median(s1))
                
            elif(A1==ssta3 and A2==ssta2):
                row,col=i+0,j+4
                s1=[int(B[row,col-1,0]),int(B[row,col,0]),int(B[row+1,col-1,0]),int(B[row+3,col-3,0])
                ,int(B[row+4,col-4,0]),int(B[row+4,col-3,0]),int(B[row+1,col-1,0]),int(B[row+3,col-3,0])
                ,int(B[row+1,col-1,0]),int(B[row+3,col-3,0]),int(B[row+1,col-2,0]),int(B[row+3,col-2,0])]
                R1=int(statistics.median(s1))
                
            elif(A1==ssta3 and A2==ssta4):
                row,col=i+0,j+4
                s1=[int(B[row,col-1,0]),int(B[row,col,0]),int(B[row+1,col-1,0]),int(B[row+3,col-3,0])
                ,int(B[row+4,col-4,0]),int(B[row+4,col-3,0]),int(B[row+1,col-1,0]),int(B[row+3,col-3,0])
                ,int(B[row+1,col-1,0]),int(B[row+3,col-3,0]),int(B[row+2,col-3,0]),int(B[row+2,col-1,0])]
                R1=int(statistics.median(s1))
                
            elif(A1==ssta4 and A2==ssta1):
                row,col=i+2,j+0
                s1=[int(B[row+1,col,0]),int(B[row,col,0]),int(B[row,col+1,0]),int(B[row,col+3,0])
                ,int(B[row,col+4,0]),int(B[row-1,col+4,0]),int(B[row,col+1,0]),int(B[row,col+3,0])
                ,int(B[row,col+1,0]),int(B[row,col+3,0]),int(B[row-1,col+1,0]),int(B[row+1,col+3,0])]
                R1=int(statistics.median(s1))
                
            elif(A1==ssta4 and A2==ssta2):
                row,col=i+2,j+0
                s1=[int(B[row+1,col,0]),int(B[row,col,0]),int(B[row,col+1,0]),int(B[row,col+3,0])
                ,int(B[row,col+4,0]),int(B[row-1,col+4,0]),int(B[row,col+1,0]),int(B[row,col+3,0])
                ,int(B[row,col+1,0]),int(B[row,col+3,0]),int(B[row-1,col+2,0]),int(B[row+1,col+2,0])]
                R1=int(statistics.median(s1))
                  
            elif(A1==ssta4 and A2==ssta3):
                row,col=i+2,j+0
                s1=[int(B[row+1,col,0]),int(B[row,col,0]),int(B[row,col+1,0]),int(B[row,col+3,0])
                ,int(B[row,col+4,0]),int(B[row-1,col+4,0]),int(B[row,col+1,0]),int(B[row,col+3,0])
                ,int(B[row,col+1,0]),int(B[row,col+3,0]),int(B[row-1,col+3,0]),int(B[row+1,col+1,0])]
                R1=int(statistics.median(s1))
                
            
            #for second plate 
            
            if(B1==ssta1 and B2==ssta2):
                row,col=i+0,j+0
                s2=[int(B[row+1,col,1]),int(B[row,col,1]),int(B[row+1,col+1,1]),int(B[row+3,col+3,1])
                ,int(B[row+4,col+4,1]), int(B[row+3,col+4,1]),int(B[row+1,col+1,1]),int(B[row+3,col+3,1])
                ,int(B[row+1,col+1,1]),int(B[row+3,col+3,1]),int(B[row+1,col+2,1]),int(B[row+3,col+2,1])]
                R2=int(statistics.median(s2))
            
               
            elif(B1==sstb1 and B2==sstb3):
                row,col=i+0,j+0
                s2=[int(B[row+1,col,1]),int(B[row,col,1]),int(B[row+1,col+1,1]),int(B[row+3,col+3,1])
                ,int(B[row+4,col+4,1]), int(B[row+3,col+4,1]), int(B[row+1,col+1,1]),int(B[row+3,col+3,1])
                ,int(B[row+1,col+1,1]),int(B[row+3,col+3,1]),int(B[row+1,col+3,1]),int(B[row+3,col+1,1])]
                R2=int(statistics.median(s2))
            
            elif(B1==sstb1 and B2==sstb4):
                row,col=i+0,j+0
                s2=[int(B[row+1,col,1]),int(B[row,col,1]),int(B[row+1,col+1,1]),int(B[row+3,col+3,1])
                ,int(B[row+4,col+4,1]), int(B[row+3,col+4,1]),int(B[row+1,col+1,1]),int(B[row+3,col+3,1])
                ,int(B[row+1,col+1,1]),int(B[row+3,col+3,1]),int(B[row+2,col+1,1]),int(B[row+2,col+3,1])]
                R2=int(statistics.median(s2))
                
            elif(B1==sstb2 and B2==sstb1):
                row,col=i+0,j+2
                s2=[int(B[row,col-1,1]),int(B[row,col,1]),int(B[row+1,col,1]),int(B[row+3,col,1])
                ,int(B[row+4,col,1]),int(B[row+4,col+1,1]),int(B[row+1,col,1]),int(B[row+3,col,1])
                ,int(B[row+1,col,1]),int(B[row+3,col,1]),int(B[row+1,col-1,1]),int(B[row+3,col+1,1])]
                R2=int(statistics.median(s2))
                
            elif(B1==sstb2 and B2==sstb3):
                row,col=i+0,j+2
                s2=[int(B[row,col-1,1]),int(B[row,col,1]),int(B[row+1,col,1]),int(B[row+3,col,1])
                ,int(B[row+4,col,1]),int(B[row+4,col+1,1]),int(B[row+1,col,1]),int(B[row+3,col,1])
                ,int(B[row+1,col,1]),int(B[row+3,col,1]),int(B[row+1,col+1,1]),int(B[row+3,col-1,1])]
                R2=int(statistics.median(s2))
                
            elif(B1==sstb2 and B2==sstb4):
                row,col=i+0,j+2
                s2=[int(B[row,col-1,1]),int(B[row,col,1]),int(B[row+1,col,1]),int(B[row+3,col,1])
                ,int(B[row+4,col,1]),int(B[row+4,col+1,1]),int(B[row+1,col,1]),int(B[row+3,col,1])
                ,int(B[row+1,col,1]),int(B[row+3,col,1]),int(B[row+2,col-1,1]),int(B[row+2,col+1,1])]
                R2=int(statistics.median(s2))
                
            elif(B1==sstb3 and B2==sstb1):
                row,col=i+0,j+4
                s2=[int(B[row,col-1,1]),int(B[row,col,1]),int(B[row+1,col-1,1]),int(B[row+3,col-3,1])
                ,int(B[row+4,col-4,1]),int(B[row+4,col-3,1]),int(B[row+1,col-1,1]),int(B[row+3,col-3,1])
                ,int(B[row+1,col-1,1]),int(B[row+3,col-3,1]),int(B[row+1,col-3,1]),int(B[row+3,col-1,1])]
                R2=int(statistics.median(s2))
                
            elif(B1==sstb3 and B2==sstb2):
                row,col=i+0,j+4
                s2=[int(B[row,col-1,1]),int(B[row,col,1]),int(B[row+1,col-1,1]),int(B[row+3,col-3,1])
                ,int(B[row+4,col-4,1]),int(B[row+4,col-3,1]),int(B[row+1,col-1,1]),int(B[row+3,col-3,1])
                ,int(B[row+1,col-1,1]),int(B[row+3,col-3,1]),int(B[row+1,col-2,1]),int(B[row+3,col-2,1])]
                R2=int(statistics.median(s2))
                
            elif(B1==sstb3 and B2==sstb4):
                row,col=i+0,j+4
                s2=[int(B[row,col-1,1]),int(B[row,col,1]),int(B[row+1,col-1,1]),int(B[row+3,col-3,1])
                ,int(B[row+4,col-4,1]),int(B[row+4,col-3,1]),int(B[row+1,col-1,1]),int(B[row+3,col-3,1])
                ,int(B[row+1,col-1,1]),int(B[row+3,col-3,1]),int(B[row+2,col-3,1]),int(B[row+2,col-1,1])]
                R2=int(statistics.median(s2))
                
            elif(B1==sstb4 and B2==sstb1):
                row,col=i+2,j+0
                s2=[int(B[row+1,col,1]),int(B[row,col,1]),int(B[row,col+1,1]),int(B[row,col+3,1])
                ,int(B[row,col+4,1]),int(B[row-1,col+4,1]),int(B[row,col+1,1]),int(B[row,col+3,1])
                ,int(B[row,col+1,1]),int(B[row,col+3,1]),int(B[row-1,col+1,1]),int(B[row+1,col+3,1])]
                R2=int(statistics.median(s2))
                
            elif(B1==sstb4 and B2==sstb2):
                row,col=i+2,j+0
                s2=[int(B[row+1,col,1]),int(B[row,col,1]),int(B[row,col+1,1]),int(B[row,col+3,1])
                ,int(B[row,col+4,1]),int(B[row-1,col+4,1]),int(B[row,col+1,1]),int(B[row,col+3,1])
                ,int(B[row,col+1,1]),int(B[row,col+3,1]),int(B[row-1,col+2,1]),int(B[row+1,col+2,1])]
                R2=int(statistics.median(s2))
                  
            elif(B1==sstb4 and B2==sstb3):
                row,col=i+2,j+0
                s2=[int(B[row+1,col,1]),int(B[row,col,1]),int(B[row,col+1,1]),int(B[row,col+3,1])
                ,int(B[row,col+4,1]),int(B[row-1,col+4,1]),int(B[row,col+1,1]),int(B[row,col+3,1])
                ,int(B[row,col+1,1]),int(B[row,col+3,1]),int(B[row-1,col+3,1]),int(B[row+1,col+1,1])]
                R2=int(statistics.median(s2))
                
            
            #for third slide
            
            if(C1==sstc1 and C2==sstc2):
                row,col=i+0,j+0
                s3=[int(B[row+1,col,2]),int(B[row,col,2]),int(B[row+1,col+1,2]),int(B[row+3,col+3,2])
                ,int(B[row+4,col+4,2]), int(B[row+3,col+4,2]),int(B[row+1,col+1,2]),int(B[row+3,col+3,2])
                ,int(B[row+1,col+1,2]),int(B[row+3,col+3,2]),int(B[row+1,col+2,2]),int(B[row+3,col+2,2])]
                R3=int(statistics.median(s3))
            
               
            elif(C1==sstc1 and C2==sstc3):
                row,col=i+0,j+0
                s3=[int(B[row+1,col,2]),int(B[row,col,2]),int(B[row+1,col+1,2]),int(B[row+3,col+3,2])
                ,int(B[row+4,col+4,2]), int(B[row+3,col+4,2]), int(B[row+1,col+1,2]),int(B[row+3,col+3,2])
                ,int(B[row+1,col+1,2]),int(B[row+3,col+3,2]),int(B[row+1,col+3,2]),int(B[row+3,col+1,2])]
                R3=int(statistics.median(s3))
            
            elif(C1==sstc1 and C2==sstc4):
                row,col=i+0,j+0
                s3=[int(B[row+1,col,2]),int(B[row,col,2]),int(B[row+1,col+1,2]),int(B[row+3,col+3,2])
                ,int(B[row+4,col+4,2]), int(B[row+3,col+4,2]),int(B[row+1,col+1,2]),int(B[row+3,col+3,2])
                ,int(B[row+1,col+1,2]),int(B[row+3,col+3,2]),int(B[row+2,col+1,2]),int(B[row+2,col+3,2])]
                R3=int(statistics.median(s3))
                
            elif(C1==sstc2 and C2==sstc1):
                row,col=i+0,j+2
                s3=[int(B[row,col-1,2]),int(B[row,col,2]),int(B[row+1,col,2]),int(B[row+3,col,2])
                ,int(B[row+4,col,2]),int(B[row+4,col+1,2]),int(B[row+1,col,2]),int(B[row+3,col,2])
                ,int(B[row+1,col,2]),int(B[row+3,col,2]),int(B[row+1,col-1,2]),int(B[row+3,col+1,2])]
                R3=int(statistics.median(s3))
                
            elif(C1==sstc2 and C2==sstc3):
                row,col=i+0,j+2
                s3=[int(B[row,col-1,2]),int(B[row,col,2]),int(B[row+1,col,2]),int(B[row+3,col,2])
                ,int(B[row+4,col,2]),int(B[row+4,col+1,2]),int(B[row+1,col,2]),int(B[row+3,col,2])
                ,int(B[row+1,col,2]),int(B[row+3,col,2]),int(B[row+1,col+1,2]),int(B[row+3,col-1,2])]
                R3=int(statistics.median(s3))
                
            elif(C1==sstc2 and C2==sstc4):
                row,col=i+0,j+2
                s3=[int(B[row,col-1,2]),int(B[row,col,2]),int(B[row+1,col,2]),int(B[row+3,col,2])
                ,int(B[row+4,col,2]),int(B[row+4,col+1,2]),int(B[row+1,col,2]),int(B[row+3,col,2])
                ,int(B[row+1,col,2]),int(B[row+3,col,2]),int(B[row+2,col-1,2]),int(B[row+2,col+1,2])]
                R3=int(statistics.median(s3))
                
            elif(C1==sstc3 and C2==sstc1):
                row,col=i+0,j+4
                s3=[int(B[row,col-1,2]),int(B[row,col,2]),int(B[row+1,col-1,2]),int(B[row+3,col-3,2])
                ,int(B[row+4,col-4,2]),int(B[row+4,col-3,2]),int(B[row+1,col-1,2]),int(B[row+3,col-3,2])
                ,int(B[row+1,col-1,2]),int(B[row+3,col-3,2]),int(B[row+1,col-3,2]),int(B[row+3,col-1,2])]
                R3=int(statistics.median(s3))
                
            elif(C1==sstc3 and C2==sstc2):
                row,col=i+0,j+4
                s3=[int(B[row,col-1,2]),int(B[row,col,2]),int(B[row+1,col-1,2]),int(B[row+3,col-3,2])
                ,int(B[row+4,col-4,2]),int(B[row+4,col-3,2]),int(B[row+1,col-1,2]),int(B[row+3,col-3,2])
                ,int(B[row+1,col-1,2]),int(B[row+3,col-3,2]),int(B[row+1,col-2,2]),int(B[row+3,col-2,2])]
                R3=int(statistics.median(s3))
                
            elif(C1==sstc3 and C2==sstc4):
                row,col=i+0,j+4
                s3=[int(B[row,col-1,2]),int(B[row,col,2]),int(B[row+1,col-1,2]),int(B[row+3,col-3,2])
                ,int(B[row+4,col-4,2]),int(B[row+4,col-3,2]),int(B[row+1,col-1,2]),int(B[row+3,col-3,2])
                ,int(B[row+1,col-1,2]),int(B[row+3,col-3,2]),int(B[row+2,col-3,2]),int(B[row+2,col-1,2])]
                R3=int(statistics.median(s3))
                
            elif(C1==sstc4 and C2==sstc1):
                row,col=i+2,j+0
                s3=[int(B[row+1,col,2]),int(B[row,col,2]),int(B[row,col+1,2]),int(B[row,col+3,2])
                ,int(B[row,col+4,2]),int(B[row-1,col+4,2]),int(B[row,col+1,2]),int(B[row,col+3,2])
                ,int(B[row,col+1,2]),int(B[row,col+3,2]),int(B[row-1,col+1,2]),int(B[row+1,col+3,2])]
                R3=int(statistics.median(s3))
                
            elif(C1==sstc4 and C2==sstc2):
                row,col=i+2,j+0
                s3=[int(B[row+1,col,2]),int(B[row,col,2]),int(B[row,col+1,2]),int(B[row,col+3,2])
                ,int(B[row,col+4,2]),int(B[row-1,col+4,2]),int(B[row,col+1,2]),int(B[row,col+3,2])
                ,int(B[row,col+1,2]),int(B[row,col+3,2]),int(B[row-1,col+2,2]),int(B[row+1,col+2,2])]
                R3=int(statistics.median(s3))
                  
            elif(C1==sstc4 and A2==sstc3):
                row,col=i+2,j+0
                s3=[int(B[row+1,col,2]),int(B[row,col,2]),int(B[row,col+1,2]),int(B[row,col+3,2])
                ,int(B[row,col+4,2]),int(B[row-1,col+4,2]),int(B[row,col+1,2]),int(B[row,col+3,2])
                ,int(B[row,col+1,2]),int(B[row,col+3,2]),int(B[row-1,col+3,2]),int(B[row+1,col+1,2])]
                R3=int(statistics.median(s3))
                
            
            # Fuzzy switching algorithm
            T1=80
            T2=135
            ma,mb,mc=0,0,0
            if(aij<T1):
                ma=0
            elif(aij>=T1 and aij<((T1+T2)/2)):
                ma=2*((aij-T1/T2-T1)*(aij-T1/T2-T1))
            elif(aij>=(T1+T2)/2 and T2>aij):
                ma=1-2*((aij-T1/T2-T1)*(aij-T1/T2-T1))
            elif(aij>=T2):
                ma=1
            
            C[i+2,j+2,0]=tha+ma*(R1-tha)        
            
            #for second slide
            if(bij<T1):
                mb=0
            elif(bij>=T1 and bij<((T1+T2)/2)):
                mb=2*((bij-T1/T2-T1)*(bij-T1/T2-T1))
            elif(bij>=(T1+T2)/2 and T2>bij):
                mb=1-2*((bij-T1/T2-T1)*(bij-T1/T2-T1))
            elif(bij>=T2):
                mb=1
            
            C[i+2,j+2,1]=thb+mb*(R1-thb)
                
            # for third slide
            if(cij<T1):
                mc=0
            elif(cij>=T1 and cij<((T1+T2)/2)):
                mc=2*((cij-T1/T2-T1)*(cij-T1/T2-T1))
            elif(cij>=(T1+T2)/2 and T2>cij):
                mc=1-2*((cij-T1/T2-T1)*(cij-T1/T2-T1))
            elif(cij>=T2):
                mc=1
            
            C[i+2,j+2,2]=thc+mc*(R1-thc)

            
            
no_noisy_3=0
p=512
for i in range(p):
    for j in range(p):
        for k in range(3):
            if(C[i,j,k]==255 or C[i,j,k]==0):
                no_noisy_3+=1
                    
cv2.imshow("Filtered Image :",C)
print(C)
print("Final Noise ",no_noisy_3)


            
cv2.waitKey()
cv2.destroyAllWindows()