#convert tif back to images

#import torch
import numpy as np
#from PIL import Image
#import matplotlib.pyplot as plt
import cv2
img3 = cv2.imread("/home/user/Desktop/ind/masks/mask2tif.tif", cv2.IMREAD_UNCHANGED)
print(img3.shape)
img3[img3 > 253] = 0
img3[img3 > 0] = 255
img3 = img3.astype('uint8')
cv2.imwrite(f"/home/user/Desktop/ind/masks/mask2png.png",img3)
'''
img1 = cv2.imread("/home/user/Desktop/ind/f1.tif", cv2.IMREAD_UNCHANGED) 
img2 = cv2.imread("/home/user/Desktop/bestp2nm2.tif", cv2.IMREAD_UNCHANGED)  # .IMREAD_GRAYSCALE
print(img1.shape,img2.shape)
req_resolution = 0.65   #in meters
curr_resolution = 0.65
req_dim = 256
#dim1 = int((req_dim *req_resolution)//curr_resolution)
dim1 = 256
#print(dim1)
h = img1.shape[0]
hr = h // dim1
w = img1.shape[1]
wr =w // dim1
#c = img2.shape[2]
print(f"{hr}, {wr}, Total images = {hr*wr}")
img1 = (img1)/2047
img1 = img1 * 256
img1 = img1.astype('uint8')
cv2.imwrite(f"/home/user/Desktop/ind/p1.png",img1) #.astype('uint8')
img2 = (img2)/2047
img2 = img2 * 256
img2 = img2.astype('uint8')
cv2.imwrite(f"/home/user/Desktop/ind/bestp2nm2.png",img2) #.astype('uint8')
print(img2.dtype)
'''

