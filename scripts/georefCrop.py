# For crooping tiff images

import random
import rasterio
from rasterio.windows import Window

with rasterio.open('/home/user/Desktop/ind/f1.tif') as src:

    # The size in pixels of your desired window
    xsize, ysize = 17920,14080
    #15104,32256

    # Generate a random window origin (upper left) that ensures the window 
    # doesn't go outside the image. i.e. origin can only be between 
    # 0 and image width or height less the window width or height
    xmin, xmax = 0, src.width - xsize
    ymin, ymax = 0, src.height - ysize
    xoff, yoff = 0, 0
    #xoff, yoff = random.randint(xmin, xmax), random.randint(ymin, ymax)

    # Create a Window and calculate the transform from the source dataset    
    window = Window(xoff, yoff, xsize, ysize)
    transform = src.window_transform(window)
    # Create a new cropped raster to write to
    profile = src.profile
    profile.update({
        'height': xsize,
        'width': ysize,
        'transform': transform})

    with rasterio.open('/home/user/Desktop/ind/resgeoref.tif', 'w', **profile) as dst:
        # Read the data from the window and write it to the output raster
        dst.write(src.read(window=window))
