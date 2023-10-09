#convert png back to tiff file, Refrence tif should be provided

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.plot import show
import cv2
#Input png image, to convert as geotiff

img = rio.open('/home/user/Desktop/ind/resulttest2.png')
print(img.shape)
img = img.read([1])  #,2,3
img = img.astype('uint16')
print(img.shape)
#img = cv2.imread(f"/home/user/Desktop/ind/result01.png", cv2.IMREAD_COLOR)
#show(img) #shows true color

# Input image for coordinate reference
with rio.open('/home/user/Desktop/ind/f1.tif') as naip:
    #open georeferenced.tif for writing
    print(naip.crs) #img.shape[0]
    with rio.open(
        '/home/user/Desktop/ind/georef_newdataset1_filtered900.tif',
        'w',
        driver='GTiff',
        count=1,                
        height=img.shape[1],
        width=img.shape[2],
        dtype=img.dtype,
        crs=naip.crs,
        transform=naip.transform,
        ) as dst:
            dst.write(img)
'''
with rio.open('/home/user/temporal_datasets/whubcd/georeferenced.tif') as limg:
        show(limg) #1 band only shows
        show(limg.read([1,2,3])) #shows true color
        #resample so pixels overlap with reference
        limg = limg.read(out_shape=(3,naip.shape[0],naip.shape[1]),
                         resampling=Resampling.nearest)
        with rio.open('resampled.tif','w', 
                     driver='GTiff',
                     count=limg.shape[0],
                     height=limg.shape[1],
                     width=limg.shape[2],
                     dtype=limg.dtype,
                     crs=naip.crs,
                     transform=naip.transform,
                     ) as dst:
            dst.write(limg)
           
'''
