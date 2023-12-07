import matplotlib.pyplot as plt
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
import csv
import random
import math
import re

def clusterSort(centers, hueWeight, lumWeight):
    """
    method to sort the centers of each cluster by hue and luminence from darkest to lightest.

    :params centers: 20x3 int array
        each row represents one cluster, with it's respective three columns being it's bgr value
    :params hueWeight: int
        order of precedence to sort the centers upon, ie how much should hue be taken into account. Varies from bread type to bread type
    :params lumWeight: int
        how much should luminence be taken into account when sorting? Varies from bread type to bread type
    :return int array
        the sorted order of the clusters (ie centers[ordering[0]]) is the darkest center

        
    """
    centerHue = {}      #define new dictionary (key value pair set)
    i = 0
    for center in centers:
        b, g, r = center
        lum = lumin(b, g, r)
        h, s, v = rgb_to_hsv(r, g, b)
        if abs(v - 100) < 10 or v < 35: #ignoring white and black
            h = 300
        centerHue[i] = hueWeight*h + lumWeight*lum
        i += 1
    sortedCenters = dict(sorted(centerHue.items(), key = lambda item: item[1]))  #sort by hue value from smallest to largest
    ordering = np.array(list(sortedCenters.keys()))   #return the set of keys as an array
    return ordering




def checkCluster (partCrust,K, hueWeight, lumWeight,imgContoured,canvas, selK):

    #blur the image first to obscure the air pockets stuff
    img = cv2.bilateralFilter(partCrust,9,150,150)

    #k-means clustering
    Z = img.reshape((-1,3))     #flatten 2D image array
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers) 
    ordering = clusterSort(centers, hueWeight, lumWeight) # array holding indexes of darkest -> lightest hues/luminences in the centers array (ie centers[ordering[0]]
                                            # is the darkest hue/luminence)

    #VLJ: I have added this so you can get the image from the clusters
    labelsFlat = label.flatten()
    intermediateImage = centers[labelsFlat]
    blur1 = intermediateImage.reshape((img.shape))

    #VLJ: This is a better way to set the darkest selK clusters to white
    
    #selK = 10
    clusteredImageR = np.zeros_like(blur1).reshape((-1, 3))
    colouredImage = np.zeros_like(blur1).reshape((-1, 3))
    colList = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]
    for i in range(0, selK):
        #b, g, r = centers[ordering[i]]
        #lum = lumin(b, g, r)
        #h, s, v = rgb_to_hsv(r, g, b)
        #print(str(i) + "," + str(ordering[i]) + "," + str(lum) + ", [" + str(h) + "," + str(s) + "," + str(v) + "]")
        clusteredImageR[labelsFlat == ordering[i]] = [255, 255, 255]
        colouredImage[labelsFlat == ordering[i]] = [colList[i][2], colList[i][1], colList[i][0]]
    kmeansImg = clusteredImageR.reshape((blur1.shape)) # for masking
    colouredImage = colouredImage.reshape((blur1.shape))
    imTest = kmeansImg

    #processing kmeansImg to find the contours
    greyscale = cv2.cvtColor(kmeansImg, cv2.COLOR_BGR2GRAY)    #convert to grey   
    contours = cv2.findContours(greyscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]  #find contours

    #contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True) #sorting the contours from smallest to largest
    #print(cv2.contourArea(contoursSorted[len(contoursSorted)-1]))

    for i in range(0, len(contours)):
        cnt = contours[i]
        cvArea =  cv2.contourArea(cnt)
        if cvArea > 3000:    #excluding contours with small area, removes the little air holes which have been captured
            if cvArea < 1000000:
                cv2.drawContours(canvas, [cnt], -1, (255,255,255), thickness=cv2.FILLED)  #draw onto the canvas
                cv2.drawContours(imgContoured, [cnt], -1, (255, 0, 0), thickness=3)
            else:
                cv2.drawContours(imgContoured, [cnt], -1, (0, 0, 255), thickness=3)
                print(str(i) + "," + str(cvArea))
        #else:
        #    cv2.drawContours(imgContoured, [cnt], -1, (0, 0, 255), thickness=3)

    return imTest, colouredImage #canvas




def checkLeftOver (crust, canvas, img):
    imgContoured = np.copy(img)
    
    #get the edge of the bread
    contoursCrust = cv2.findContours(crust, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    crustEdge = np.zeros_like(crust, np.uint8)
    for cntCrust in contoursCrust:
        #draw the edge of the bread
        cv2.drawContours(crustEdge, [cntCrust], -1, (255, 255, 255), thickness=1)

    kernel = np.ones((5,5),np.uint8)
    outlineCrust = 6
    dilation = cv2.dilate(crustEdge,kernel,iterations = outlineCrust)
    crustCanvas = cv2.bitwise_or(canvas,dilation)

    #find all the contours on the crust that are not thought to be the crust
    canvas = np.zeros_like(crust, np.uint8)
    contours, hierarchy = cv2.findContours(crustCanvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    font = cv2.FONT_HERSHEY_SIMPLEX

    contHier = zip(contours,hierarchy)
    contoursSorted = sorted(contHier, key=lambda x: cv2.contourArea(x[0])) #sorting the contours from smallest to largest
    
    if cv2.contourArea(contoursSorted[len(contoursSorted)-1][0]) > 1000000: # this if statement accounts of the case where there is a contours 
                                                                         # that goes around the entire crust (if i filled this the whole slice would be white)
        cv2.drawContours(canvas, [contoursSorted[len(contoursSorted)-1][0]],-1, (255, 255, 255), thickness=cv2.FILLED)
        cv2.drawContours(imgContoured, [contoursSorted[len(contoursSorted)-1][0]],-1, (0, 255, 0), thickness=3)
        cv2.drawContours(canvas, [contoursSorted[len(contoursSorted)-2][0]],-1, (0, 0, 0), thickness=cv2.FILLED)
        cv2.drawContours(imgContoured, [contoursSorted[len(contoursSorted)-2][0]],-1, (0, 255, 0), thickness=3)

    for i in range(0, len(contoursSorted) - 2):
        cnt = contoursSorted[i][0]
        hier = contoursSorted[i][1]
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        #only get the contours that are on the edge of the breads
        if ((area < 10000) and (hier[3] == 0)):
            cv2.drawContours(imgContoured, [cnt], -1, (255, 0, 255), thickness=-1)
            cv2.drawContours(canvas, [cnt], -1, (255,255,255), thickness=-1)
        elif (hier[3] == 0) and (area > 1000000):#should never go here
            print("WHY:" + str(i) + "," + str(hier)[1:-1] + "," + str(area))


    return imgContoured,canvas

def blobDetection (img,crust,K, hueWeight, lumWeight,selK):
    
    #divide the crust to left, right, top and bottom
    contours, _ = cv2.findContours(crust,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    image_out = {}
    crustList = ["L","R","T","B"]
    for i in crustList:
        image_out[i] = np.zeros_like(crust, np.uint8) #initialise black background
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        #cv2.drawContours(image_out, [cnt], -1, (255,0,0), 5)

        #get the rough crustSize
        crustSize = 0.25 #0.15
        crustWidth = math.ceil(crustSize * w)
        crustHeight = math.ceil(crustSize * h)
        #print(str(crustWidth) + "," + str(crustHeight))

        #left crust
        cv2.rectangle(image_out["L"],(x,y),(x + crustWidth, y + h),(255,255,255),-1)
        #right crust
        cv2.rectangle(image_out["R"],(x + w - crustWidth,y),(x + w, y + h),(255,255,255),-1)
        #top crust
        cv2.rectangle(image_out["T"],(x,y),(x + w, y + crustHeight),(255,255,255),-1)
        #bottom crust
        cv2.rectangle(image_out["B"],(x,y + h),(x + w, y + h - crustHeight),(255,255,255),-1)
    
    #Find the contour for the crust area
    imgContoured = np.copy(img)
    canvas = np.zeros_like(crust, np.uint8)
    imCrust = -1
    colorCrustList = []
    for i in crustList:
        print(i)
        maskCrust = cv2.bitwise_and(image_out[i], crust)
        imCrust, colourCrust = checkCluster(cv2.bitwise_and(img, img, mask = maskCrust),K, hueWeight, lumWeight,imgContoured,canvas,selK) 
        colorCrustList.append(colourCrust) 

    #Do sanity check on the crust, so the crust area that's in between the edges and the found crust are counted as crust
    #Check that the found crust is really on the edge of the bread and not inside
    imgContoured, canvas = checkLeftOver (crust, canvas, img)

    return imgContoured, canvas, colorCrustList

"""
Get the possible area for the crust
"""
def getOnlyCrust (img):
    #Get the outline of the bread
    #change colour space
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #thresholding
    mask = cv2.inRange(mask, (0, 45, 25), (27, 255, 255))

    #Find the contour for the crust area
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    image_out = np.zeros_like(mask, np.uint8) #initialise black background

    area_min = 100000
    n_contour = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area >= area_min):
            cv2.drawContours(image_out, [cnt], -1, (255,255,255), -1)

    crust = mask

    #Get the mask out area
    image_processed = cv2.bitwise_and(img, img, mask = crust)

    return crust,image_processed

def getEroded(img):
    """
    method to get only the possible area for where the crust could be in a slice of bread

    :param img: 3d int array
        the original image in bgr colour coding
    :return image_processed: 3d int array
        the image with the only the crust region painted on a black background


    """
    #Get the outline of the bread
    #change colour space
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #thresholding
    mask = cv2.inRange(mask, (0, 45, 25), (27, 255, 255))

    #Find the contour for the crust area
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    image_out = np.zeros_like(mask, np.uint8) #initialise black background

    area_min = 100000
    n_contour = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area >= area_min):
            cv2.drawContours(image_out, [cnt], -1, (255,255,255), -1)

    #Erode the contour
    kernel = np.ones((5,5),np.uint8)
    crustSize = 50

    erosion = cv2.bitwise_not(cv2.erode(image_out,kernel,iterations = crustSize))
    crust = cv2.bitwise_and(image_out, erosion)

    #Get the mask out area
    image_processed = cv2.bitwise_and(img, img, mask = crust)

    #image_processed = cv2.dilate(crustCanvas, kernel, iterations=crustSize)

    return image_processed


"""
Write information about the crust
"""
def writeCrustData(file, area, widths, lum, h, s, v):
    with open ("outputData/crustData.csv", 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        data = [file, area, widths[2], widths[0], widths[1], widths[3], lum, h, s, v]
        writer.writerow(data)


def getContourSkeleton(file, canvas):

        crust = skimage.img_as_bool(skimage.color.rgb2gray(canvas))
        skel, distance = skimage.morphology.medial_axis(crust, return_distance=True)#get the distance transform and skeleton
        dist_output = distance * skel #get the middle distance
        width_dist = dist_output * 2 #get the actual distance

        #calculate the max, min and average distance
        max_width = np.max(width_dist[skel])
        min_width = np.min(width_dist[skel])
        average_width = np.mean(width_dist[skel])
        std_width = np.std(width_dist[skel])

        #output the skeleton and distance
        io.imsave("outputImages/crust/skeleton/"+file, dist_output)

        #print(str(min_width),str(max_width),str(average_width),str(std_width))
       
        return min_width, max_width, average_width, std_width


def crustWidth(crustCanvas, img, scale):
    """

    NOT NEEDED ANYMORE, CRUST SKELETONISE DOES BETTER JOB


    method to determine the thickness of the crust at each point on the edge of the bread

    works by using the "getOnlyCrust" method to get the annular region that the crust should be within. thresholds this image to find a mask, takes the second 
    beiggest contour as the inner ring of the annulus (which is the border between the possible crust area and the interior bread area). Then steps along this 
    contours 100 pixels at a time, constructing a vector out of each step. Calculates the vector perpendicular to this and produce a new region, which should   
    be the region which contains one little 100 pixel wide slice of the crust. Uses this to crop the crustCanvas img, thereby slicing the crust into small regions.
    Uses a distance to map to figure out the width of this piece of crust. Does this for all pieces and stores the information in an array.

    :params img: 3d int array
        the original image in bgr colour coding
    :params crustCanvas: 3d int array
        the canvas produced by the 'findCrust' method (the crust painted in white on a black background)


    """
    widths = []

    crustMask = getEroded(img)


    hsv = cv2.cvtColor(crustMask, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, (0,0,0), (255,255,60))
    thresh = cv2.threshold(thresh, 125, 255, cv2.THRESH_BINARY_INV)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]  #find contours
    contSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

    interiorCont = contSorted[len(contSorted)-2]    #should be the interior ring of the annulus, which is where the interior of the bread starts


    for i in range(0, len(interiorCont)-1):     #now stepping along this contour 50 pixels at a time
        if i%50 == 0:    #if i is a multiple of 50
            try:            #try catch needed for the case where we hit the edge of the contour, ie point i+100 is not in the contour
                #cv2.circle(img, (interiorCont[i][0]), radius=5, color=(255,0,0), thickness=-1)
                x,y = interiorCont[i][0]        #getting the pixel coordinates at this point on the contour, i
                y = abs(y-crustCanvas.shape[0])     #transforming y into cartesian coordinates
                endX, endY = interiorCont[i+50][0]     #figuring out the pixel coordinates of the next point in the contour, i+100
                endY = abs(endY - crustCanvas.shape[0])     #tranforming y into cartesian coordinates
                deltaX = endX - x       #figuring out the change in x and y from point i to i+100, ie producing direction vector for the line through them
                deltaY = endY - y

                #apply anticlockwise roation of 90 degrees to the two components of the direction vector, producing a vector perpendicular to the line from point i to i+50
                perpendicularX = -deltaY    
                perpendicularY = deltaX


                xFinal = x
                yFinal = y
                t = 5

                while(xFinal < 2480 and yFinal < 1650 and xFinal > 0 and yFinal > 0):
                    xFinal = x + t*perpendicularX
                    yFinal = y + t*perpendicularY
                    t += 1

                #cv2.line(img, [x,abs(y-crustCanvas.shape[0])], [x+deltaX, abs(y+deltaY-crustCanvas.shape[0])], (0,255,0), thickness=5)
                #cv2.line(img, [x,abs(y-crustCanvas.shape[0])], [xFinal, abs(yFinal - crustCanvas.shape[0])], (0,255,0), thickness=5)


                #using the direction and perpendicular vector to construct a new region, which should just be a little slice of the crust
                sliceContour = np.array([[x, abs(y-crustCanvas.shape[0])], [endX, abs(endY - crustCanvas.shape[0])], 
                            [xFinal + deltaX, abs(yFinal + deltaY - crustCanvas.shape[0])], [xFinal, abs(yFinal - crustCanvas.shape[0])]])
                
                
                """
                mask = np.zeros(img.shape, dtype=np.uint8)
  
                cv2.fillPoly(mask, pts=[sliceContour], color=(255,255,255))              #filling the sliced contour region in with white on the black mask

                croppedImg = cv2.bitwise_and(crustCanvas, mask)     #bitwise and operation croped the crustCanvas image so that only the masked area (region bounded
                                                                    #by slice contour) is showing.
                #"""


                x,y,w,h = cv2.boundingRect(sliceContour)
                if not (max(h,w)/min(h,w)) < 3:
                    croppedImg = crustCanvas[max(1,y):y+h, max(1,x):x+w]
                    greyscale = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
                    dist = cv2.distanceTransform(greyscale, cv2.DIST_L2, 3)     #distance tranform, produces a map which calculates the distance from every point in the contour to the 
                                                                                #nearest point outside the contour. Information stored in an array
                    minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(dist)           #gets the values and locations of the maximum and minimum held within the array
                    if not (2*maxVal/scale) in widths:
                        widths.append(2*maxVal/scale)
                    """plotting work
                    cv2.circle(croppedImg, maxLoc, 3, (200,0,0), -1)
                    plot_images("fa", [img, crustCanvas, croppedImg, dist])
                    #"""


            except:
                continue

    return widths

def findCrust(file, img, selK, hueWeight, lumWeight, canvas, scale, crustImage):
    """

    DEOSN'T ACTUALLY DO THIS ANYMORE BECAUSE CRUST DETECTION HAS BEEN DONE WITH OTHER METHOD, THIS IS REALLY JUST TO GET THE WIDTHS NOW AND WRITE TO OUTPUT

    method to identify the crust on a piece of bread

    works by using k-means clustering to seperate the colours in the image. Then sorts the cluster by hue and luminence, and takes the first 6 clusters
    to be the crust. Produces a new image, kmeansImg, where the crust clusters are coded to white, and everything else is black. Then finds the contours
    of this new image and then filters out the smallest before painting onto a blank canvas. The interior of the canvas is then blacked out using a mask 
    produced by the "getOnlyCrust" method. Finally, the contours of the canvas are painted back onto the original image (for visual aid), summed up to 
    give total area, and this image plus the canvas are returned.

    :params img: 3d int array
        represents the image in bgr colour coding
    :params hueWeight: int
        how much the method which sorts the centers should sort based on hue (as opposed to luminence)
    :params lumWeight: int
        how much the method which sorts the centers should sort based on luminence (as opposed to hue)
    :params crustMask: 3d int array
        represents the image of the mask produced by "getOnlyCrust" method in bgr colour coding
    :params crustImage: 3d int array
        represents the crust only part of the image in bgr colour coding
    :params crustBW: 3d int array
        represents the crust only part of the image in black and white colour coding
    :return imgContoured: 3d int array
        a copy of the original image with the contours found to be the crust painted on to it in blue
    :return canvas: 3d int array
        a black and white image which shoes the crust in white on a black background


    """
    imgContoured = np.copy(img)

    #k-means clustering on the crustImage
    Z = crustImage.reshape((-1,3))     #flatten 2D image array
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z,selK,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers) 
    ordering = clusterSort(centers, hueWeight, lumWeight) # array holding indexes of darkest -> lightest hues/luminences in the centers array (ie centers[ordering[0]]
                                            # is the darkest hue/luminence)

    #doing this process just so I can write the average luminence, hue, saturation, and value of the crust to the csv.
    crustHues = []
    crustSats = []
    crustVals = []
    crustLums = []

    for i in range(0,selK):
        b,g,r = centers[ordering[i]]
        h,s,v = rgb_to_hsv(r,g,b)
        crustHues.append(h)
        crustSats.append(s)
        crustVals.append(v)
        crustLums.append(lumin(b,g,r))

    crustHue = np.mean(crustHues)
    crustSat = np.mean(crustSats)
    crustVal = np.mean(crustVals)
    crustLum = np.mean(crustLums)

    #processing canvas to find the contours
    canvasGreyscale = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)    #convert to grey   
    canvasThresh = cv2.threshold(canvasGreyscale, 125, 255, cv2.THRESH_BINARY)[1]   #threshold
    canvasContours = cv2.findContours(canvasThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  #find contours
    canContSorted = sorted(canvasContours, key=lambda x: cv2.contourArea(x))

    cv2.drawContours(imgContoured, canvasContours, -1, (255, 0, 0), thickness=3)


    area = 0
    if cv2.contourArea(canContSorted[len(canContSorted)-1]) > 1000000:
        area = (cv2.contourArea(canContSorted[len(canContSorted)-1]) - cv2.contourArea(canContSorted[len(canContSorted)-2]))/math.pow(scale,2)
    else:
        for i in range(0, len(canvasContours)):
            area += cv2.contourArea(canvasContours[i])/math.pow(scale,2)

    min, max, avg, std = getContourSkeleton(file, canvas)

    widths = [min/scale,max/scale,avg/scale,std/scale]

    writeCrustData(file, area, widths, crustLum, crustHue, crustSat, crustVal)


"""
Get the definite area of the crust
"""
def thickCrust (img, K, selK, hueWeight, lumWeight):
    imgContoured = np.copy(img)

    #k-means clustering on the crustImage
    Z = img.reshape((-1,3))     #flatten 2D image array
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers) 
    ordering = clusterSort(centers, hueWeight, lumWeight) # array holding indexes of darkest -> lightest hues/luminences in the centers array (ie centers[ordering[0]]
                                            # is the darkest hue/luminence)

    #VLJ: I have added this so you can get the image from the clusters
    labelsFlat = label.flatten()
    intermediateImage = centers[labelsFlat]
    blur1 = intermediateImage.reshape((img.shape))

    clusteredImageR = np.zeros_like(blur1).reshape((-1, 3))
    for i in range(0, K):
        if i < selK:
          clusteredImageR[labelsFlat == ordering[i]] = [255,255,255]#[colList[i][2], colList[i][1], colList[i][0]]
        #else:
        #  clusteredImageR[labelsFlat == ordering[i]] = [0, 0, 0]
    kmeansImg = clusteredImageR.reshape((blur1.shape)) # for masking

    #processing kmeansImg to find the contours
    greyscale = cv2.cvtColor(kmeansImg, cv2.COLOR_BGR2GRAY)    #convert to grey   
    contours = cv2.findContours(greyscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]  #find contours

    colList = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]
    counter = 0
    contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x)) #sorting the contours from smallest to largest
    canvas = np.zeros_like(kmeansImg, np.uint8)
    smallerContour = []
    for i in range(0, len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        cvArea =  cv2.contourArea(contours[i])
        if cvArea > 3000:    #excluding contours with small area, removes the little air holes which have been captured
            if cvArea < 1000000:
                cv2.drawContours(canvas, contours, i, (255,255,255), thickness=cv2.FILLED)  #draw onto the canvas
                cv2.drawContours(imgContoured, contours, i, (255, 0, 0), thickness=3)
            else:
                cv2.putText(
                            img = imgContoured,
                            text = str(i),
                            org = (x, y),
                            fontFace = cv2.FONT_HERSHEY_DUPLEX,
                            fontScale = 0.5,
                            color = (125, 246, 55),
                            thickness = 3
                            )
                if (counter > 0):
                    smallerContour.append(contours[i])
                else:
                    cv2.drawContours(imgContoured, contours, i, colList[counter], thickness=3)#(0, 255, 0)
                    cv2.drawContours(canvas, contours, i, (255,255,255), thickness=cv2.FILLED)
                counter = counter + 1
                print(str(i) + "," + str(counter) + "," + str(cv2.contourArea(contours[i])))
        #else:
        #    cv2.drawContours(imgContoured, contours, i, (0, 0, 255), thickness=3)
    
    for cnt in smallerContour:
        cv2.drawContours(canvas, [cnt], -1, (0,0,0), thickness=cv2.FILLED)  #draw onto the canvas
        cv2.drawContours(imgContoured, [cnt], -1, (0, 0, 255), thickness=3)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    return imgContoured, canvas


"""
    method dealing with identifying crust information from the bread (call all the other crust related method)
"""
def identifyCrust (file, img, scale, resetCrust):

    selK = 6 #10
    K = 20
    hueWeight = 1
    lumWeight = 0
    
    if (resetCrust == 1):
        mask, breadMask = getOnlyCrust(img)
        
        blob, blobCanvas,colourCrustList = blobDetection (img,mask,K, hueWeight, lumWeight,selK)
        i = 0

        cv2.imwrite("outputImages/crust/crust/"+file, blob)
        #cv2.imwrite("crustCanvas/"+file, blobCanvas)

    canvasName = "crustCanvas/" + file
    canvas = cv2.imread(canvasName)

    canvasGrey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    crustOnly = cv2.bitwise_and(img, img, mask = canvasGrey)
    #cv2.imwrite("outputImages/crust/justCrust"+file, crustOnly)   

    findCrust(file, img, selK, hueWeight, lumWeight, canvas, scale, crustOnly)
