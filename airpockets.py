from typing import List
import matplotlib.pyplot as plt
import matplotlib
import cv2
from skimage import io, measure
import random
import os
import re
import numpy as np
from imageCropper import crustCrop
import csv
import math
from general import lumin
from general import rgb2linDot
from general import rgb_to_hsv
from general import plot_images





def clusterSort(centers):
    """
    Method to sort the clusters' centers from darkest luminence to brightest
    
    :param centers:  20x3 integer array
            Each row is the center of a cluster with the 3 columns being the b,g,r channels of the center
    :return: A list indexes in the centers array, which represent the order from lowest to highest luminence

    """
    clusterPairs = {}   #define a new dictionary (key-value pair set)
    i = 0
    for cluster in centers:                    #iterate over clusters, calculate the luminence of each, insert into dictionary
        b,g,r = cluster

        if b < 50 and g < 50 and r < 50:
            clusterPairs[i] = 300
        else:
            clusterPairs[i] = lumin(b,g,r)
        i+=1
    sortedCenters = dict(sorted(clusterPairs.items(), key = lambda item: item[1]))  #sort by luminence value from smallest to largest
    return np.array(list(sortedCenters.keys()))   #return the set of keys as an array



def removeCrust(img, canvas):
    """
    Method to crop out the crust on a piece of bread. 

    :param img: HxWx3 integer array
        The image being worked on
    :param canvas: HxWx3 integer array
        A mask for the area of the crust, the area to be cropped out
    :return croppedImg: HxWx3 integer array
        The image with the crust cropped out
    """

    ker = np.ones((10,10), dtype="uint8") #dilation operation just to smooth out the edge
    canvas = cv2.dilate(canvas, ker, 1)

    croppedImg = cv2.bitwise_or(img, canvas)

    #now going to clean up the cropped image by taking the largest contour (should be the internal bread), thereby removing isolted bits of crust
    mask = np.ones(img.shape, dtype=np.uint8)
    mask.fill(255)

    grey = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)[1]

    outerContours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]    #finding contours
    outerContours = sorted(outerContours, key=lambda x: cv2.contourArea(x))                  #sorting them from smallest to largest

    sliceContour = [outerContours[len(outerContours)-1]]            #this is an array of points which constitute the contour that borders the bread slice

    channelCount = img.shape[2]                                     #this is colour guf, just so it works with images that have different numbers of channels
    ignoreMaskColour = (0,)*channelCount                            #though we could have assumed RGB encoding so it doesn't matter

    cv2.fillPoly(mask, sliceContour, ignoreMaskColour)              #filling the contour in with black on the white mask

    finalImg = cv2.bitwise_or(croppedImg, mask)                          #producing the cropped image with some sort of cv2 magic
      
    return finalImg


def writeScoreData(file, scores):
    for i in range(0, len(scores)-1):
        data = [file, i, scores[i]]
        with open ("outputData/scoreData.csv", 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)


def writeAirpocketData(file, img, clusterNo, score, clusterLum, h, s, v, scale):
    """
    Write information about the airpockets, area and luminence, to a CSV file
    Works by recieving the image from colour coding a single cluster onto a white background, 
    Then finds the contours on this image to find the area of each airhole within the cluster

    :params file: string
            The name of the image file being worked on
    :params img: 1664x2496x3 integer array 
            With the first two representing the pixel position, and the third representing the b,g,r channels of the pixel
    :params clusterNo: integer
            The index of the current cluster IN THE CENTERS array.
    :params clusterLum: integer
            The luminence of the current cluster

    """
    contouredImg = np.copy(img)

    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(greyscale, 225, 255, 1)[1]


    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(contouredImg, contours, -1, (0,0,0), thickness=cv2.FILLED)

    for i in range(0, len(contours)-1):
        if not cv2.contourArea(contours[i]) == 0:

            data = [clusterNo, score, clusterLum, h, s, v, i, cv2.contourArea(contours[i])/math.pow(scale,2)]
            with open ("outputData/airpocketData/" + file.split('.')[0] + ".csv", 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    


def writeClusterData(file, centers):
    """
    Write information about luminence and hue of the spread of clusters (ie max, min, average etc) that contain airpockets to a CSV file.

    :params file: str 
            The name of the image file being worked on
    :params centers: 20x3 integer array
            Each row represents a cluster 1 through 20, the 3 columns are the b,g,r values of that cluster
    :params ordering: integer array
            The ordering of the indexes of the centers array from lowest luminence to highest.
            

    """
    lums = []
    hues = []
    sats = []
    vals = []
    for i in range(0,len(centers)-1):
        b,g,r = centers[i]
        lums.append(lumin(b,g,r))
        hues.append(rgb_to_hsv(r,g,b)[0])
        sats.append(rgb_to_hsv(r,g,b)[1])
        vals.append(rgb_to_hsv(r,g,b)[2])

    lums = np.asarray(lums)
    hues = np.asarray(hues)
    sats = np.asarray(sats)
    vals = np.asarray(vals)
    
    data = [file, np.min(lums), np.max(lums), np.mean(lums), np.median(lums), np.std(lums), 
            np.min(hues), np.max(hues), np.average(hues), np.median(hues), np.std(hues),
            np.min(sats), np.max(sats), np.average(sats), np.median(sats), np.std(sats),
            np.min(vals), np.max(vals), np.average(vals), np.median(vals), np.std(vals)]

    with open("outputData/clusterData.csv", 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    
    return lums, hues, sats, vals



def getStandardisedColour(score):
    """
    Method to get the colour of the cluster

    :param score: int
        the metric by which to colour the cluster by. This is only called score because I coloured them first by score, in the current version the colouring
        is actually done by luminence, so the name is deceiving.


    """
    cmap = matplotlib.cm.get_cmap('hsv')    #using hsv colour map

    #this is a little bit of fine-tuning to emphasize the differences in the middle range of the data, where most of it sits. This all has to change when
    #colouring by a different attribute.
    percentile = (score-30)/35
    if percentile < 0.2:    #clipped at 0.2 because the only red on the image I want to be the maximum, not the minumum
        percentile = 0.2
    elif percentile > 1:
        percentile = 1

    r,g,b,a = cmap(percentile)

    return 255*b,255*g,255*r



def overlay(file, img, scores, clusterImgs, subtitles, scale):
    """
    Method to plot the clusters on top of the original bread slice image.
    
    :param file: String
        the name of the image file
    :param img: HxWx3 integer array
        The original image of the bread slice
    :param scores: int[]
        The array of values used to get the colour of the image. While this is called score, in the current version the colouring is actually being done by 
        luminance, so the name is misleading 
    :param clusterImgs: 12xHxWx3 integer array
        The array containing the images of all the clusters painted on white backgrounds
    :param subtitles: int[]
        Array of cluster numbers which were found to be airpockets (neccessary to produce stratified version of image)


    """

    combined = np.copy(img)
    overlayed = []
    overlayed.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    troubleshoot = np.copy(img)

    for i in range(0, len(clusterImgs)-1):
        #overlayed.append(np.copy(img)) 

        greyscale = cv2.cvtColor(clusterImgs[i], cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(greyscale, 200, 255, 1)[1]
    
        contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

        colour = getStandardisedColour(scores[i])

        for k in range(0,len(contours)):
            #cv2.drawContours(overlayed[i], contours, k, colour, thickness=cv2.FILLED)

            cv2.drawContours(combined, contours, k, colour, thickness=2)

            #if cv2.contourArea(contours[k])/math.pow(scale, 2) > 300:
            #    cv2.drawContours(troubleshoot, contours, k , colour, thickness = 3)
            #    plt.imshow(troubleshoot)
            #    plt.show()

        #overlayed[i] = cv2.cvtColor(overlayed[i], cv2.COLOR_BGR2RGB)    #changing colour to rgb space for plotting 



    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)    #changing colour back to rgb space for plotting
    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(combined)
    fig.tight_layout()

    #plotting the colour bar
    sm = plt.cm.ScalarMappable(cmap='hsv')
    sm.set_clim(vmin=25, vmax=60)       #note max and min change when colouring by a different attribute
    plt.colorbar(sm, shrink=0.3, aspect=20*0.5)

    ax.axis("off")

    #saving figure and reopening is clumsy, but for some reason the cropping didn't seem to work on the figure itself, 
    #so saved as a working image and cropped and passed that instead
    plt.savefig("outputImages/workingImage.JPG", format='jpg')
    croppedImg = cv2.imread("outputImages/workingImage.JPG")
    croppedImg = croppedImg[500:1500, 200:1900]
   
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15), gridspec_kw={'height_ratios': [1,1.5]}) 

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(croppedImg, cv2.COLOR_BGR2RGB))

    ax1.axis("off")
    ax2.axis("off")


    plt.tight_layout()
    plt.savefig("outputImages/airpockets/airpockets/" + file.split('.')[0] + ".JPG", format='jpg')
    plt.close(f)




def clusterRankAndSize(image):
    """
    method to find the average size of the contours in an image, and the total number of contours
    
    :params image: WxHx3 int array
        the image to find contours within. Only works if this image is the contours painted on a white background
    :return len(contours): int
        the number of contours in the image
    :return area/(len(contours)+1): int
        the average area of the contours 
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY_INV)[1]    #thresholding assumes that the contours are painted on a white background
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

    area = 0
    for contour in contours:
        area += cv2.contourArea(contour)

    return len(contours), area/(len(contours)+1)



def airpockets(file, img, canvas, numberPercentile, sizePercentile, denominator, scale):
    """
    method to find the airpockets in a slice of bread.

    :params file: String
        this is the name of the image file
    :params img: LxWx3 int array
        this is the array representing the image in bgr encoding
    :params canvas: LxWx3 int array
        this array represents the image of the crust in white on a black background. Needed by the remove crust method, which uses the canvas to crop out the crust
    :params numberPercentile: int
        the percentile for the number of airpockets in a cluster that we want to exclude, varies from bread to bread
    :params sizePercentile: int
        the percentile for the average size of the airpockets ina cluster that we want to exclude, varies from bread to bread
    :params denominator: int
        used to make the cutoff for bread score lower or higher, varies from bread to bread.


    """
    img = removeCrust(img, canvas)  #call to remove cust method, which returns the slice of bread with the crust cropped out

    #K Means clustering algorithm
    K = 20
    Z = img.reshape((-1,3))     #flatten 2D image array
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    # turn label and center into numpty arrays
    label = np.uint8(label) # Nx1 array with N = number of pixels, with respective cluster that they belong too
    centers = np.uint8(center) #Kx3 array with K being the number of clusters, and each cluster has a b,g,r value

    #VLJ: I have added this so you can get the image from the clusters
    labelsFlat = label.flatten()
    intermediateImage = centers[labelsFlat]
    blur1 = intermediateImage.reshape((img.shape))
    kmeansImg = blur1


    imagesArray = []        #will consist of the individual images for each, stratified cluster
    subtitles = []          #will become a list of cluster numbers which have been included as airpockets
    clusteredImage = np.copy(blur1).reshape((-1, 3))       #going to be the image combined from all the clusters   


    #filling the the luminence, hues, saturations, and values of the cluster centers
    lums, hues, sats, vals = writeClusterData(file, centers) 
    
    
    #filling in arrays with the average size of each airpocket and the total number of airpockets pertained in each cluster. 
    #Useful for excluding clusters which have too many contours and each contour is too small to be an airpocket
    count = []
    size = []

    #iterating over the clusters. Need to create a blank white canvas for each cluster, and then paint the clusters onto this canvas and pass to clusterRankAndSize, which finds the contours
    #on this canvas and figures and returns how many and average size
    for i in range(0,K):    
        whiteCluster = np.empty_like(blur1).reshape(-1,3)
        whiteCluster.fill(255)

        whiteCluster[labelsFlat == i] = (0,0,0)

        count.append(clusterRankAndSize(whiteCluster.reshape(blur1.shape))[0])     
        size.append(clusterRankAndSize(whiteCluster.reshape(blur1.shape))[1])    

    count = np.asarray(count)   #casting to numpy array
    size = np.asarray(size)

    #finding thresholds for the number of airpockets and the average size of airpockets.
    numberThresh = np.percentile(count, numberPercentile)
    sizeThresh = np.percentile(size, sizePercentile)            #MADE A MISTAKE HERE FOR MOST OF THE PROJECT, DID PERCENTILE OF COUNT ARRAY NOT SIZE, SO sizeThresh
                                                                #WOULD HAVE BEEN MASSIVE

    lumCol = [] #this array is needed for the purposes of colouring the clusters in the output image

    index = 0      
    for i in range(0,K):    #iterating over the clusters, calculating the score for each

        b,g,r = centers[i]
        lum = lumin(b,g,r)
        h,s,v = rgb_to_hsv(r,g,b)

        lumScore, hueScore, satScore, valScore = (0,0,0,0)  #setting default value to 0
#        if lum > np.median(lums):               #it only matters if the cluster is above the median. If it's significantly below then doesn't matter                                  
        lumScore = lum - np.median(lums)
#        if h > np.median(hues):
        hueScore = h - np.median(hues)
#        if s < np.median(sats):
        satScore = np.median(sats) - s
#        if v > np.median(vals):
        valScore = v - np.median(vals)

        score = lumScore + 3*hueScore + 5*satScore + valScore           #saturation is highly weighted because a) the values tend to be smaller
                                                                        #                                      b) it seems to be the best predictor of airpockets, with hue coming in second   
        bench = (np.std(lums) + 3*np.std(hues) + 5*np.std(sats) + np.std(vals))/denominator  #the denominator here essentially decides how strict to be with the cutoff.
                                                                                             #on some types of bread it doesn't work very well to filter based on airpocket size or rank, so have to use
                                                                                             #scoring more heavily and have to be more precise with cutoff       


        if count[i] > numberThresh  or size[i] < sizeThresh:           #typically, can automatically exclude the clusters with both very high rank but very small average area, these don't generally constitue real airpockets
            score = 100                                                 #could just continue, but for troubleshooting helps to see which clusters where excluded on this basis

        #print("Cluster " + str(i) + ": score = " + str(score) + ", bench = " + str(bench) + ", numberClusters = " + str(count[i]) + ", size = " + str(size[i]) +    #printing for troubleshooting
        #", luminence = " + str(lum) + ", hue = " + str(h) + ", saturation = " + str(s) + ", value = " + str(v))

        if score <= bench:
            imagesArray.append(np.empty_like(blur1).reshape(-1, 3))           #appending a copy of the image
            imagesArray[index].fill(255)

            #painting cluster onto both individual and combined image
            imagesArray[index][labelsFlat == i] = getStandardisedColour(lum)           
            clusteredImage[labelsFlat == i] = getStandardisedColour(lum)

            #have to reshape and convert colour to rgb for plotting
            imagesArray[index] = imagesArray[index].reshape(blur1.shape)
            imagesArray[index] = cv2.cvtColor(imagesArray[index], cv2.COLOR_BGR2RGB)

            writeAirpocketData(file, imagesArray[index], i, score, lum, h, s, v, scale)

            subtitles.append(i) #appending i to subtitles array so that we know which clusters were taken as airpockets
            index += 1          #need to keep track of the size of the images array (which is how many clusters have been taken as airpockets)
    
            lumCol.append(lum)  #building an array of the luminence of airpocket clusters for colouring
    
    clusteredImage = clusteredImage.reshape(blur1.shape)
    clusteredImage = cv2.cvtColor(clusteredImage, cv2.COLOR_BGR2RGB)

    imagesArray.append(clusteredImage)

    overlay(file, img, lumCol, imagesArray, subtitles, scale)


