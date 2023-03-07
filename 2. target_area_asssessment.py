# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:47:10 2023

@author: alexd
"""

import gc
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rs
from rasterio.plot import show
from matplotlib.colors import ListedColormap


#Plotting non-permanent water
path = r"4th_trial\\water_history.tif"

with rs.open(path, 'r') as ds:
    img = ds.read() 
    
perma_water = np.zeros_like(img)
perma_water[np.where((img > 0) & (img < 96))] = 1
perma_water = perma_water.reshape((1766, 1825))
    


fig, ax = plt.subplots()
im = ax.matshow(perma_water, cmap='gray')

# add a color bar below the image
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
                    
plt.show()


#Plotting masks
mask = "530P"
X_path = r"4th_trial\\swe_mask\\4_Inputs\\Masking_" + mask + "\\training_" + mask + ".npy"   #4-input data

X = []; y_label = []
X = np.load(X_path)

fig, ax = plt.subplots()

colorarray2 = ["#5EBDAA","#2C8598"]
   #cmap1 = ListedColormap(colorarray1)
cmap2 = ListedColormap(colorarray2)
masked_img = np.ma.masked_where(X[1,:,:,0] == -1, X[1,:,:,0])

im = ax.matshow(masked_img, cmap = cmap2)

# add a color bar below the image
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
                    
plt.show()



#Intersection
label = r"4th_trial\\label_2021.npy" #complete Snow Extent
y_label = np.load(label)


x_y = X[1,:,:,0] - y_label[1] 
perm = np.where(((img > 0) & (img < 96)))

idx = np.array(np.where(x_y != 0))
idx = np.transpose(idx)
perm = np.transpose(perm)
perm = perm[:,1:3]
aset = set([tuple(x) for x in idx])
bset = set([tuple(x) for x in perm])
x = np.array([x for x in aset & bset])

intersection = tuple(e for e in np.transpose(x))
  
loc =  np.zeros_like(img).reshape((1766, 1825))
loc[intersection] = 1


fig, ax = plt.subplots()
im = ax.matshow(loc, cmap='gray')

# add a color bar below the image
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
                    
plt.show()









