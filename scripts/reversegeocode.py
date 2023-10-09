#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:23:00 2020

@author: dataeaze
"""

import numpy as np
import logging
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
import cv2
import torch
import rasterio
import pyproj
import reverse_geocoder as rg
import pprint
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import csv
def reverseGeocode(coordinates):
	result = rg.search(coordinates)
	#pprint.pprint(result)
	print(result)

wgs84 = pyproj.Proj(projparams = 'epsg:4326')
InputGrid = pyproj.Proj(projparams = 'epsg:32643')
transformer = pyproj.Transformer.from_crs("epsg:32643", "epsg:4326")


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

def scores2(img1,th,ptr1):
    global count1,count2
    #cv2.imwrite(f"/home/user/Desktop/ind/filtered_mask/orignal_w{ptr1}.png",img1)
    ret, thresh = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    #mask1 = img1.astype(bool)
    print(len(contours))
    #maskr = np.zeros((18006,14259))
    lsttemp1 = []
    result_corr = []
    for i in range(len(contours)):
        lsttemp2 = []
        for j in range(len(contours[i])):
            y_cord = contours[i][j][0][0]
            x_cord = contours[i][j][0][1]
            #print(y_cord,x_cord)
            lsttemp2.append([x_cord,y_cord])
        lsttemp1.append(lsttemp2)
    for coordinates1 in lsttemp1:
        polygon1 = np.array(coordinates1)
        maskt1 = torch.from_numpy(polygon2mask(img1.shape, polygon1))
        maskt2 = polygon2mask(img1.shape, polygon1)
        maskt2.dtype = np.uint8
        count1 += 1
        if torch.count_nonzero(maskt1).item()<th:
            continue
        else:
            result_corr.append((coordinates1[0][0],coordinates1[0][1]))
            #maskr += (maskt2*255)
            count2 += 1
            #print("Hi")
    #cv2.imwrite(f"/home/user/Desktop/ind/zzzz_resulttest2.png",maskr)
    return result_corr


if __name__ == "__main__":
    print("Loading ............... ")
    img1 = cv2.imread("/home/user/Desktop/ind/resulttest2.png")
    img1 = img1[:,:,0]
    th = 900
    print(img1.shape)
    coordinates = scores2(img1,th,0)
    #print(coordinates)
    print(count2)
    data_list = []
    with rasterio.open('/home/user/Desktop/ind/masks/mask2tif.tif') as map_layer:
        for j in range(len(coordinates)):
            pixels2coords = map_layer.xy(coordinates[j][0],coordinates[j][1])  #input px, py
            #print(pixels2coords)
            x1, y1 = pixels2coords[0],pixels2coords[1]
            xn1,yn1 = transformer.transform(x1, y1)
            #print(xn1,yn1)
            new_coordinates = (xn1,yn1)
            #reverseGeocode(new_coordinates)
            geolocator = Nominatim(user_agent="application")
            reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
            location = reverse(new_coordinates, language='en')
            #print(location.raw)
            data_list.append(location.raw)
    
    header = data_list[0].keys()
    csv_file = 'sample.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for data in data_list:
            writer.writerow(data)
        print(f'CSV file "{csv_file}" has been created successfully.')
