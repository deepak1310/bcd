# merge results obtained by model forward pass
import cv2
import torch
import numpy as np
import time
#new1 = np.zeros((17920,14080,3))
new1 = np.zeros((18006,14259,1))
count = 0
for i in range(70):
    print(i)
    if i%10==0:
        time.sleep(1)
    for j in range(55):
        temp1 = cv2.imread(f"/home/user/Desktop/ind/resulttest2/{count}per.png", cv2.IMREAD_COLOR)
        #print(temp1.shape)
        new1[256*i:256*(i+1),256*j:256*(j+1),0] = temp1[:,:,0]
        count += 1
        #new1[256*i:256*(i+1),256*j:256*(j+1),0] = temp1[:,:,0]*0
        #new1[256*i:256*(i+1),256*j:256*(j+1),1] = temp1[:,:,0]*0
        #new1[256*i:256*(i+1),256*j:256*(j+1),2] = temp1[:,:,0]

cv2.imwrite("/home/user/Desktop/ind/resulttest2.png",new1)
print(count)
        
