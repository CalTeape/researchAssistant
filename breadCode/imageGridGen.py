import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
import seaborn as sns
from airpockets import removeCrust
from imageCropper import crustCrop
import re
import math
import matplotlib
from general import getScale
from general import resize
from typing import List
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from skimage import io, measure
import random
import os
import re
import numpy as np
import csv
import numpy as np
from pathlib import Path
import math
import json
import os
from operator import itemgetter


def plot_images(images: List[np.ndarray], figure_title:str = "", subplot_titles:List[str]=[]):
    """
    Plot a series of images in a grid using matplotlib and subplots
    Useful to show interpretations of different filters in conv-nets

    :param images: List[np.ndarray]
            The list of images to be plotted
            Images (items of the array) should be square, etc.. and ar passed directly to imshow
            The list itself can be any length (even non-square)

    :param figure_title: str
            The title of the entire figure

    :param subplot_titles: List[str]
            Title each image. Defaults to empty array.
            If empty, no titles are added

    :param cmap: str
            The colour map to use for the images


    """


    total_cols = int(len(images))
    total_rows =1

    print(subplot_titles)

    fig = plt.figure(figsize=(total_cols*6,8))
    fig.suptitle(figure_title)
    for i in range(0,len(images)):
        ax = fig.add_subplot(total_rows, total_cols, i+1)
        ax.imshow(images[i])
        ax.axis("off")
        if i < len(subplot_titles):
            ax.set_title(subplot_titles[i], fontsize=8)

    #plotting the colour bar
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    #cax = plt.axes([0.1, 0.1, 0.6, 0.075])
#
    #sm = plt.cm.ScalarMappable(cmap='hsv')
    #sm.set_clim(vmin=25, vmax=60)       #note max and min change when colouring by a different attribute
    #plt.colorbar(sm, shrink=0.1, aspect=20*0.5, orientation="horizontal", cax=cax)

    #plt.show()
    plt.savefig("outputImages/crust/crustGridView/crust/" + subplot_titles[0].split('(')[0] + ".JPG", format='jpg', dpi=600)



def main():

    dictX = {        
            "FBF50" : [],       
            "FBF5": [],
            "FBF25" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "FBPI-20": [],
            "LF25" : [],
            "PF50" : [],
            "PF5" : [],
            "PF25" : [],
            "PPI5" : [],
            "WF" : [],
            "Control" : [],
            "PGFBF-30 " : [],
            "GFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 " : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],  
    }

    breadTypes = dictX.keys()
    print(breadTypes)

    dir = "outputImages/crust/justCrust/"

    images = os.listdir(dir)

    for type in breadTypes:
        #if not re.search("Control", type):
        #        continue
        imagesArray = []
        subtitles = []
        print(type)
        for image in images:
            if re.search(type, image):
                #print(image)
                subtitles.append(image)
                #images.remove(image)
                image = cv2.imread(dir + image)
                image = np.array(image)
                #image = image[int(image.shape[0]/2):image.shape[0]-150, 0:image.shape[1]-300]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imagesArray.append(image)

        for image in subtitles:
                images.remove(image)
        
        plot_images(imagesArray, type, subtitles)

main()
