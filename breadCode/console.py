import matplotlib.pyplot as plt
import cv2
from skimage import io, measure
import numpy as np
import os
from imageCropper import crustCrop
#from crust import findCrust
#from crust import getOnlyCrust
from airpockets import airpockets
from general import adjustContrast
from general import getScale
import random
import csv
import re
from general import resize
from crustVer import identifyCrust
from pathlib import Path
from crustVer import crustWidth
from crustVer import findCrust


def getK(filename):
    """
    Method to return the optimum parameters to use for crust detection

    :params filename: String
        The name of the image file, says what type of bread it is
    
    :return the optimum number of clusters to use in k means
    :return the weighting for hues when sorting clusters
    :return the weighting for luminence when sorting clusters
    

    """
    if re.search("FBF50", filename) or re.search("LF25", filename) or re.search("PF50", filename):
        return 20, 0, 1
    elif re.search("PPI", filename):
        return 20, 0, 1
    elif re.search("FBPI", filename):  # good for the most part
        return 20, 1, 2
    elif re.search("FBF5", filename):  #fava bean 5 (difficulty)
        return 20, 1, 0
    elif re.search("FBF25", filename):  #fava bean 25   (good for the most part)
        return 20, 0 ,1
    elif re.search("PF5", filename):    #pea flour  (difficulty, tends to capture airholes aswell)
        return 20, 0, 1
    elif re.search("WF", filename):     #white bread (average)
        return 20, 2, 1
    else:
        return 20, 1, 1


def getScoring(filename):
    """
    Method to return the optimal parameters to use for airpocket detection

    :params filename: String
        Name of the image file, says what type of bread it is

    :return kernel size for median blur
    :return the percentile to use for the number of contours cutoff in airpocket cluster selection
    :return the percentile to use for the size of contours threshold in airpocket cluster selection
    :return the denominator to use when calculating the scoring benchmark in airpocket cluster selection


    """
    if re.search("Control1", filename):
        return 3, 95, 20, 2
    if re.search("Control", filename):
        return 3, 95, 35, 3      
    if re.search("PFBF-30", filename):
        return 3, 85, 0, 5
    if re.search("PGFBF-30", filename):
        return 3, 80, 0, 5
    if re.search("GFBF", filename):
        return 3, 90, 0, 10
    if re.search("FBF40 ", filename):  #the space after captures the FBF-30 (1).JPGs
        return 3, 85, 5, 5
    if re.search("FBF-30 ", filename):  #the space after captures the FBF-30 (1).JPGs
        return 3, 85, 5, 5
    if re.search("FBF-30", filename):   #the lack of space after captures the FBF-301.JPG
        return 3, 85, 0, 5
    if re.search("FBF-20", filename):
        return 3, 85, 0, 5
    if re.search("FBF-10", filename):
        return 5, 90, 0, 2.5
    if re.search("FBF5", filename):
        return 5, 90, 0, 5
    if re.search("FBF", filename) or re.search("FBP", filename) or re.search("LF", filename):
        return 5, 85, 0, 10
    if re.search("PF5", filename):
        return 3, 90, 0, 15
    if re.search("PF25", filename):
        return 3, 85, 0, 10 
    if re.search("PPI5\(8\)", filename) or re.search("PPI5\(9\)", filename) or re.search("PPI5\(6\)", filename) or re.search("PPI5\(7\)", filename):
        return 0, 90, 0, 15
    if re.search("PPI5", filename):
        return 0, 85, 0, 20
    if re.search("WF-100", filename):
        return 0, 70, 25, 3
    if re.search("WF", filename):
        return 0, 85, 0, 17.5

def checkFolder():
    #Create all the output folders if they don't exists, make it first
    pathList = ["outputData","outputData/airpocketData","outphotos","outputImages/airpockets","outputImages/crust","crustCanvas"]
    for pathLoc in pathList:
        Path(pathLoc).mkdir(parents=True, exist_ok=True)

def main():
    
    checkFolder()

    #"""
    with open("outputData/clusterData.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        data = ["image", 
            "min lum", "max lum", "avg lum", "median lum", "lum std", 
            "min hue", "max hue", "avg hue", "median hue", "hue std",
            "min saturation", "max saturation", "avg saturation", "median saturation", "saturation std",
            "min value", "max value", "average value", "median value", "value std"]
        writer.writerow(data)


    with open ("outputData/crustData.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        data = ["image", "area", "average width", "minimum width", "maximum width", "width std", "crust luminance", "crust hue", "crust saturation", "crust value"]
        writer.writerow(data)

    #"""
    folder = "/Users/callum/Desktop/Bread2021/photos/"
    dirList = os.listdir(folder)
    random.shuffle(dirList)

    for dir in dirList:
        if os.path.isdir(folder+dir):
            fileList = os.listdir(folder+dir+ '/') 
    
            for file in fileList:
                if not file.lower().endswith(('.jpg')) or re.search("Lens", file) or re.search("coloured", dir):
                    continue
                else:    

                    with open ("outputData/airpocketData/" + file.split('.')[0] + ".csv", 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        data = ["cluster number", "score", "luminance", "hue", "saturation", "value", "count", "area"]
                        writer.writerow(data)

                    filename = folder+dir+'/'+file
                    print(filename)
                    img = cv2.imread(filename)
                    img = resize(img)
   
                    scale = getScale(file, img)


                    img = crustCrop(img)


                    #"""Calling process for crust
                    resetCrust = 1 #1 = repeat the whole finding crust process; 0 = just run the crust width bit
                    identifyCrust (file, img, scale, resetCrust)
                    #"""

                    #"""Calling process for airpockets
                    ker, numberPercentile, sizePercentile, denominator = getScoring(file)

                    if not ker == 0:
                        img = cv2.medianBlur(img, ker)

                    img = adjustContrast(img, 1.4)

                    canvasName = "crustCanvas/" + file
                    canvas = cv2.imread(canvasName)
                    airpockets(file, img, canvas, numberPercentile, sizePercentile, denominator, scale)
                    #"""

        else: 
            continue
        

main()















