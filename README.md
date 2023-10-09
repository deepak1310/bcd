1. Tiny_model_4_CD contains traning.py and test_ondata.py for tranining and testing. For more detail read readme inside the Tiny_model_4_CD
2. Models take images of 256 x 256, whereas we have two tif files of pune you can acces them as p1ntif and p2msetif (https://drive.google.com/drive/folders/1NmEFHYcYTcXpGoOTX5ufm-Z-7xkZaeJ9?usp=drive_link)
3. Keep only intersection area of 2 images (Before and After) using qgis
4. Use pngconv.py to convert into png
5. If required do histogram matching using histogramMatch.py
6. Do image regestration using image_reg_msecalc.py (Avoid step 3, 5and 6 if you are using gdrive p1ntif and p2msetif as they are already preprocessed)
7. Use split.py to split them into 256 x 256
8. Train Model and get results
9. Merge result using resmerger.py
10. Convert result into tif by png2georef.py
