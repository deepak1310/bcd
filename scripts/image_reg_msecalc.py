

import cv2 
import numpy as np
img2 = cv2.imread("/home/user/Desktop/ind/p2.png", cv2.IMREAD_GRAYSCALE)  # TBC
img1 = cv2.imread("/home/user/Desktop/ind/p1.png", cv2.IMREAD_GRAYSCALE) #refrence
height, width = img2.shape 
print(img1.shape,img2.shape)
'''
new_img2 = img2
new_img2 = img2[11:img1.shape[0],3:img1.shape[1]]  # shift where minima occurs
cv2.imwrite('/home/user/Desktop/ind/p2mse.png',new_img2)'''

mini = 99999999999
wsize = 6000
for i in range(20):
    for j in range(20):
        x1 = img1[5000:5000+wsize,5000:5000+wsize]
        x2pp = img2[5000+i:5000+i+wsize,5000+j:5000+j+wsize]
        x2pn = img2[5000+i:5000+i+wsize,5000-j:5000-j+wsize]
        x2np = img2[5000-i:5000-i+wsize,5000+j:5000+j+wsize]
        x2nn = img2[5000-i:5000-i+wsize,5000-j:5000-j+wsize]
        epp = np.sum(np.square(x2pp - x1))/(wsize*wsize)
        epn = np.sum(np.square(x2pn - x1))/(wsize*wsize)
        enp = np.sum(np.square(x2np - x1))/(wsize*wsize)
        enn = np.sum(np.square(x2nn - x1))/(wsize*wsize)
        emin = min(epp,epn,enp,enn)
        if mini>emin:
            print("----------------------------------------------------------------------------------------------------------------------------------------------->",i,j,emin)
            mini = emin
        print(i,j,epp,epn,enp,enn)
        
        
print("Complete")
