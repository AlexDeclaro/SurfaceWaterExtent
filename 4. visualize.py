# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:38:57 2023

@author: alexd
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np

def plot(img, prev, prediction, path, i): #path = figures path, i = number of testing data.
    plt.figure(figsize=(10, 10))
    #subplot_num = 1 if prediction is None else 3
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    # オリジナル画像の描画
    masked_img = np.ma.masked_where(img == -1, img)
    masked_prev = np.ma.masked_where(prev == -1, prev)

    
    #colorarray1 = ["#ffffff", "#5EBDAA","#2C8598"]
    colorarray2 = ["#5EBDAA","#2C8598"]
    #cmap1 = ListedColormap(colorarray1)
    cmap2 = ListedColormap(colorarray2)
    
    #print(viridis(range(0,12))) to get color values

    
    ax[0].matshow(masked_img, cmap = cmap2)
    ax[0].set_axis_off()
    ax[0].set_title('Masked', y = 1)
    
    ax[1].matshow(masked_prev, cmap = cmap2)
    ax[1].set_axis_off()
    ax[1].set_title('t-1', y = 1)

    # 判定結果の描画
    ax[2].matshow(prediction, cmap = cmap2)
    ax[2].set_axis_off()
    ax[2].set_title('Prediction', y = 1)
    
    fig.subplots_adjust(hspace=1)
    plt.tight_layout()
    plt.rcParams["figure.autolayout"] = True
    plt.margins(x=0)
    
    plt.savefig("{}{}.png".format(path, i))
    
    
    plt.show()