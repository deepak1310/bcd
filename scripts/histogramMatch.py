from skimage import exposure
from skimage.exposure import match_histograms
from skimage.filters import unsharp_mask
import cv2
import matplotlib.pyplot as plt
reference = cv2.imread("/home/user/Desktop/ind/p1.png", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("/home/user/Desktop/ind/p2mse.png",cv2.IMREAD_GRAYSCALE)
#reference = data.coffee()
#image = data.chelsea()
print(reference.shape,image.shape)

matched = match_histograms(image, reference)
#cv2.imwrite('/home/user/Desktop/ind/p2msehm.png', matched)

result_1 = unsharp_mask(matched, radius=1, amount=1)
result_2 = unsharp_mask(matched, radius=5, amount=2)
result_3 = unsharp_mask(matched, radius=20, amount=1)

cv2.imwrite('/home/user/Desktop/ind/p2msehmum1.png', result_1)
cv2.imwrite('/home/user/Desktop/ind/p2msehmum2.png', result_2)
cv2.imwrite('/home/user/Desktop/ind/p2msehmum3.png', result_3)
