import cv2
  
# import Numpy
import numpy as np
  
# read a image using imread
img = cv2.imread('/home/user/Desktop/ind/p1n.png',cv2.IMREAD_UNCHANGED)
  
# creating a Histograms Equalization
# of a image using cv2.equalizeHist()
equ = cv2.equalizeHist(img)
  
# stacking images side-by-side
#res = np.hstack((img, equ))
  
cv2.imwrite('/home/user/Desktop/ind/p1nheq.png',equ)
