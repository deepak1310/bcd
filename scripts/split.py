# To split the image into 256 x 256 

import cv2
#import torch
img2 = cv2.imread("/home/user/Desktop/ind/p2mse.png", cv2.IMREAD_COLOR)  # after
img1 = cv2.imread("/home/user/Desktop/ind/p1.png", cv2.IMREAD_COLOR)  # before
label = cv2.imread("/home/user/Desktop/ind/masks/mask2png.png", cv2.IMREAD_UNCHANGED)  #labels
print(img2.shape,img1.shape,label.shape)
dim1 = 500
print(dim1)
h = img1.shape[0]
hr = h // dim1
w = img1.shape[1]
wr =w // dim1
c = img2.shape[2]
print(f"{hr}, {wr}Total images = {hr*wr}")
print(img2.dtype)
count = 0
for i in range(hr):
    for j in range(wr):
        n1 = img1[dim1*i:dim1*(i+1),dim1*j:dim1*(j+1)]
        n2 = img2[dim1*i:dim1*(i+1),dim1*j:dim1*(j+1)]
        n3 = label[dim1*i:dim1*(i+1),dim1*j:dim1*(j+1)]
        #n3 = diff_mask[dim1*i:dim1*(i+1),dim1*j:dim1*(j+1)]
        #b1 = cv2.resize(n1, (req_dim, req_dim))
        #b2 = cv2.resize(n2, (req_dim, req_dim))
        #b3 = cv2.resize(n3, (req_dim, req_dim))
        cv2.imwrite(f"/home/user/Desktop/ind/newdataset3/B/w{count}.png",n2)  #after
        cv2.imwrite(f"/home/user/Desktop/ind/newdataset3/A/w{count}.png",n1)
        cv2.imwrite(f"/home/user/Desktop/ind/newdataset3/label/w{count}.png",n3)
        #cv2.imwrite(f"/home/user/temporal_datasets/whubcd/splitedt/mask/w{count}.png",n3)
        count += 1
print("image converted = ",count)

'''

# This is not required if shited spliteed images are not required.
th = 100
for i in range(hr):
    for j in range(wr-1):
        n1 = img1[dim1*i:dim1*(i+1),dim1*j+th:dim1*(j+1)+th]
        n2 = img2[dim1*i:dim1*(i+1),dim1*j+th:dim1*(j+1)+th]
        n3 = label[dim1*i:dim1*(i+1),dim1*j+th:dim1*(j+1)+th]
        if i in [54,55,56,57]:
            print("------------>",count)
            continue
        #n3 = diff_mask[dim1*i:dim1*(i+1),dim1*j:dim1*(j+1)]
        #b1 = cv2.resize(n1, (req_dim, req_dim))
        #b2 = cv2.resize(n2, (req_dim, req_dim))
        #b3 = cv2.resize(n3, (req_dim, req_dim))
        cv2.imwrite(f"/home/user/Desktop/ind/newdataset2/B/w{count}.png",n2)  #after
        cv2.imwrite(f"/home/user/Desktop/ind/newdataset2/A/w{count}.png",n1)
        cv2.imwrite(f"/home/user/Desktop/ind/newdataset2/label/w{count}.png",n3)
        #cv2.imwrite(f"/home/user/temporal_datasets/whubcd/splitedt/mask/w{count}.png",n3)
        count += 1
print("image converted = ",count)
'''
