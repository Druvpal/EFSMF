import random
import cv2

def add_noise(img):
    row,col=img.shape
    val=0
    img = cv2.resize(img,(512,512))
    val=100
    
    #val=val
    
    for i in range(val):
        x=random.randint(0, row-1)
        y=random.randint(0, col-1)
        img[x][y]=255
    
    for j in range(val):
        x=random.randint(0, row-1)
        y=random.randint(0, col-1)
        img[x][y]=0
    
    return img

img=cv2.imread("C:\\Users\\HARSH NARAYAN PANDEY\\Downloads\\img.png",cv2.IMREAD_GRAYSCALE)
cv2.imshow("original",img)
print(img)
#cv2.imwrite("salt_lena_image",add_noise(img))
img1=add_noise(img)
#cv2.imshow("add_noise",img)
#print(img1)
cv2.waitKey()
cv2.destroyAllWindows()