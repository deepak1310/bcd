#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:23:00 2020

@author: dataeaze
"""
# Detect polygons in images and rectify accorind to Thresold area(th)
import numpy as np
import logging
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
import cv2
import torch
from torchmetrics.classification import BinaryJaccardIndex
'''
img1 = torch.from_numpy(np.array([[0,0,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]]))
img2 = torch.from_numpy(np.array([[0,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]]))
metric = BinaryJaccardIndex()
tempiou = metric(img1, img2)
print(tempiou)  # 0.5
'''

count1,count2 = 0,0

def mask2poly(mask, tolerance=1):
    
    arr = None
    if not isinstance(mask, np.ndarray): # Check if we have numpy array
        logging.error("mask must be of type numpy.array got " + str(type(mask)))
        return None, False
    elif mask.dtype == bool: # If boolean convert to integer and then float
        arr = mask.copy().astype(np.int64)
    elif issubclass(mask.dtype.type, np.integer) or \
        issubclass(mask.dtype.type, np.floating): # if integer check if binary
        arr = mask.copy().astype(np.int64)
    else:
        logging.error("only integer or floating type arrays supported")
        return None, False
    
    if not set(np.unique(arr).tolist()).issubset(set([0, 1])):
        logging.error("No values other than 0 and 1 are supported")
        return None, False
    
    arr = arr.astype(np.float64)
    level = 0.5
    contours = find_contours(arr, level,
                             fully_connected='low',
                             positive_orientation='low'
                             )
    
    polygons = [approximate_polygon(c, tolerance) for c in contours]

    # print(contours)
    # print(polygons)    
    return polygons, True 


def scores(img1, img2,th,area):
    mask1 = img1.astype(bool)
    mask2 = img2.astype(bool)
    poly1, ret1 = mask2poly(mask1)
    poly2, ret2 = mask2poly(mask2)
    #print(poly1 ,ret1)
    fp, fn, tp = 0, 0, 0
    #cnt1 = 0
    for coordinates1 in poly1:
        #print("loop 1",ret1)
        polygon1 = np.array(coordinates1)
        #coordinates = [[y,x] for [x,y] in coordinates]
        maskt1 = torch.from_numpy(polygon2mask(img1.shape, polygon1))
        if torch.count_nonzero(maskt1).item()<area:
            continue
        #cv2.imwrite(f"/home/user/Desktop/{cnt1}.png",polygon2mask(img1.shape, polygon1).astype('uint8')*255)
        #cnt1 += 1
        #print(np.unique(maskt1))
        #plt.imshow(maskt1)
        iou12, iou21 = 0, 0
        for coordinates2 in poly2:
            polygon2 = np.array(coordinates2)
            maskt2 = torch.from_numpy(polygon2mask(img2.shape, polygon2))
            metric = BinaryJaccardIndex()
            tempiou = metric(maskt1, maskt2)
            iou12 = max(iou12, tempiou)
        if iou12 >= th:
            tp += 1
        else:
            fn += 1
    for coordinates2 in poly2:
        polygon2 = np.array(coordinates2)
        #coordinates = [[y,x] for [x,y] in coordinates]
        maskt2 = torch.from_numpy(polygon2mask(img2.shape, polygon2))
        if torch.count_nonzero(maskt2).item()<area:
            continue
        #plt.imshow(maskt1)
        iou12, iou21 = 0, 0
        for coordinates1 in poly1:
            polygon1 = np.array(coordinates1)
            maskt1 = torch.from_numpy(polygon2mask(img1.shape, polygon1))
            metric = BinaryJaccardIndex()
            tempiou = metric(maskt1, maskt2)
            iou21 = max(iou21, tempiou)
        if iou21 < th:
            fp += 1

    return tp, fn, fp

def scores2(img1,th,ptr1):
    global count1,count2
    #cv2.imwrite(f"/home/user/Desktop/ind/filtered_mask/orignal_w{ptr1}.png",img1)
    ret, thresh = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    mask1 = img1.astype(bool)
    #print(contours[1].shape)
    lsttemp1 = []
    for i in range(len(contours)):
        lsttemp2 = []
        for j in range(len(contours[i])):
            y_cord = contours[i][j][0][0]
            x_cord = contours[i][j][0][1]
            lsttemp2.append([x_cord,y_cord])
        lsttemp1.append(lsttemp2)

    maskr = np.zeros((256,256))
    #poly1, ret1 = mask2poly(mask1)
    #count1 = 0
    for coordinates1 in lsttemp1:
        polygon1 = np.array(coordinates1)
        #coordinates = [[y,x] for [x,y] in coordinates]
        maskt1 = torch.from_numpy(polygon2mask(img1.shape, polygon1))
        maskt2 = polygon2mask(img1.shape, polygon1)
        maskt2.dtype = np.uint8
        #cv2.imwrite(f"/home/user/Desktop/ind/filtered_mask/w{ptr1}_{count1}.png",maskt2*255)
        #count1 += 1
        #print("--------->",torch.count_nonzero(maskt1).item())
        count1 += 1
        if torch.count_nonzero(maskt1).item()<th:
            continue
        else:
            count2 += 1
            maskr += (maskt2*255)
    cv2.imwrite(f"/home/user/temporal_datasets/LEVIRCD_train/res0.65mgray/label3/ltr{ptr1}.png",maskr)

def scores1(img1,th,ptr1):
    cv2.imwrite(f"/home/user/Desktop/ind/filtered_mask/orignal_w{ptr1}.png",img1)
    mask1 = img1.astype(bool)
    maskr = np.zeros((256,256))
    poly1, ret1 = mask2poly(mask1)
    count1 = 0
    for coordinates1 in poly1:
        #print(coordinates1.dtype)
        polygon1 = np.array(coordinates1)
        #coordinates = [[y,x] for [x,y] in coordinates]
        maskt1 = torch.from_numpy(polygon2mask(img1.shape, polygon1))
        maskt2 = polygon2mask(img1.shape, polygon1)
        maskt2.dtype = np.uint8
        cv2.imwrite(f"/home/user/Desktop/ind/filtered_mask/w{ptr1}_{count1}.png",maskt2*255)
        #count1 += 1
        #print("--------->",torch.count_nonzero(maskt1).item())
        if torch.count_nonzero(maskt1).item()<th:
            continue
        else:
            maskr += (maskt2*255)
    cv2.imwrite(f"/home/user/Desktop/ind/filtered_mask/w{ptr1}.png",maskr)

if __name__ == "__main__":
    for i in range(4005):
        #print(i)
        img1 = cv2.imread(f"/home/user/temporal_datasets/LEVIRCD_train/res0.65mgray/label/ltr{i}.png")
        #img1 = cv2.imread(f"/home/user/Desktop/ind/newdataset1/label/w{i}.png")
        #print(img1.shape)
        img1 = img1[:,:,0]
        thresold = 900
        scores2(img1,thresold,i)
    print(count1,count2)
    
