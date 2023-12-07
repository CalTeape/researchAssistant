import matplotlib.pyplot as plt
import matplotlib
import cv2
from skimage import io, measure
import skimage
import numpy as np
import os
from imageCropper import crustCrop
from general import rgb2linDot
from general import lumin
from general import rgb_to_hsv
from general import plot_images
from general import resize
import csv
import random
import math
import re



def getStandardisedColour(lum):
    """
    Method to get the colour of the cluster

    :param score: int
        the metric by which to colour the cluster by. This is only called score because I coloured them first by score, in the current version the colouring
        is actually done by luminence, so the name is deceiving.


    """
    cmap = matplotlib.cm.get_cmap('hsv')    #using hsv colour map

    #this is a little bit of fine-tuning to emphasize the differences in the middle range of the data, where most of it sits. This all has to change when
    #colouring by a different attribute.
    print(lum)
    percentile = (lum-20)/50
    if percentile < 0.2:    #clipped at 0.2 because the only red on the image I want to be the maximum, not the minumum
        percentile = 0.2
    elif percentile > 1:
        percentile = 1

    r,g,b,a = cmap(percentile)

    return 255*b,255*g,255*r


def generateCrustClusterView(file, img, canvas):

    print(img.shape)
    print(canvas.shape)

    #canvas = cv2.bitwise_not(canvas)

    #cropped = cv2.bitwise_or(img, canvas)
    cropped = cv2.bitwise_and(img, canvas)
    #hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)


    cv2.imwrite("outputImages/crust/justCrust/" + file, cropped)

    #k-means clustering on the crustImage
    Z = cropped.reshape((-1,3))     #flatten 2D image array
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z,10,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers) 


    #VLJ: I have added this so you can get the image from the clusters
    labelsFlat = label.flatten()
    intermediateImage = centers[labelsFlat]
    blur1 = intermediateImage.reshape((img.shape))
    kmeansImg = blur1


    #VLJ: This is a better way to set the darkest 3 clusters to white
    
    clusteredImageR = np.ones_like(blur1).reshape((-1, 3))
    clusteredImageR *= 0
    for i in range(len(centers)):
        b,g,r = centers[i]
        if b == 255 and g == 255 and r == 255 or b == 0 and g == 0 and r == 0:
            continue
        lum = lumin(b,g,r)
        clusteredImageR[labelsFlat == i] = getStandardisedColour(lum)

    kmeansImg = clusteredImageR.reshape((blur1.shape)) # for masking

    cv2.imwrite("outputImages/crust/clusteredCrust/" + file, kmeansImg)

def main():
    folder = "/Users/callum/Desktop/Bread2021/photos/"
    dirList = os.listdir(folder)
    random.shuffle(dirList)

    for dir in dirList:
        if os.path.isdir(folder+dir):
            fileList = os.listdir(folder+dir+ '/')
            #random.shuffle(fileList)  
    
            for file in fileList:
                if not file.lower().endswith(('.jpg')) or re.search("Lens", file) or re.search("coloured", dir):
                    continue
                else:   

                    filename = folder+dir+'/'+file
                    print(filename)
                    img = cv2.imread(filename)
                    img = resize(img)
                    img = crustCrop(img)
                    

                    """
                    #some string manipulation to make file names match 
                    filekey2 = file[0:len(file)-5] + "(" + file[len(file)-5: len(file)-4] + ")" + file[len(file)-4:]#adding brackets around number, e.g. FBF-205.JPG -> FBF-20(5).JPG
                    filekey3 = file[0:len(file)-8] + file[len(file)-7:] #removing the space in FBF 40s ie FBF 40(1) -> FBF40(1)
                    """
                    print(file)
                    canvas_name = "crustCanvas/" + file    #THIS IS WHAT NEEDS TO CHANGE TO VERONICA CRUST IMAGES
                    canvas = cv2.imread(canvas_name)
                    if canvas is None:
                        print("NEED SPECIAL CHARACTER CHECKING")
                    """
                        canvas = cv2.imread("crustCanvas/" + filekey2)
                    if canvas is None:
                        canvas = cv2.imread("crustCanvas/" + filekey3)
                    """
                    generateCrustClusterView(file, img, canvas)
        else: 
            continue
        

main()















