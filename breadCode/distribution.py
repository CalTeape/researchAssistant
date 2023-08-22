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



def imgSearch(target):
    """
    method to search through the desktop bread folder to find the image file for a given filename

    :params target: String
        the filename
    :returns img: 3d int array
        the image with the filename target (in bgr color coding)
    :returns scale: int
        the scale of the image in pixels per mm (note this is found after the image resize)
    """

    folder = "/Users/callum/Desktop/Bread2021/photos/"
    dirs = os.listdir(folder)

    #DOING SOME STRING MODIFYING SO THAT SPECIAL CHARACTERS WILL MATCH
    target = target.replace("-", "\-")
    target = target.replace("(", "\(")
    target = target.replace(")", "\)")
    
    for dir in dirs:
        if os.path.isdir(folder+dir):
            fileList = os.listdir(folder+dir+ '/')
            for file in fileList:

                if re.search(target,file):
                    if file.lower().endswith(('.jpg')):       
                        img = cv2.imread(folder+dir+'/'+file)

                        img = resize(img)
                        scale = getScale(target, img)

                        img = crustCrop(img)

                        return img, scale




def countDuplicates(arr):
    """
    method to count the duplicates in an array

    :params arr: int[]
        the input array
    :returns value: int[]
        array of values
    :returns count" int[]
        the array of counts of respective values

    e.g.
    arr = [2,2,2,3,3,2,1,8]

        -> value = [2,3,1,8]
           count = [4,2,1,1]
    """
    value = []
    count = []
    for x in arr:
        if not x in value:
            value.append(x)
            count.append(1)
        else:
            index = value.index(x)
            count[index] += 1
    
    return value, count





def avgCount():
    """
    method to plot a line graph which shows how many airpockets there are for each attribute value (ie score, luminence, etc), for each bread type (averaged over the images for that bread type)

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"]   #going to need a graph for each attribute, so iterate over each and construct graph for all bread types

    for attribute in attributes:                #going to need a graph for each attribute, so iterate over each and construct graph for all bread types
        dictX = {      
            "FBF40 ": [],                              #initialising dictionary with bread types as keys and empty arrays which will hold the data for the attributes as values
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBPI-20 " : []  
        }

        dictY = {     
            "FBF40 ": 0,                                #initialising dictionary with bread types as keys and empty arrays which will hold the number of image files for each bread type
            "FBF5": 0,
            "FBF25" : 0,
            "FBF50" : 0,
            "FBPI5" : 0,
            "FBPI10" : 0,
            "LF25" : 0,
            "PF5" : 0,
            "PF25" : 0,
            "PF50" : 0,
            "PPI5" : 0,
            "WF" : 0,
            "WF-100 " : 0,
            "Control" : 0,
            "FBF-30 " : 0,
            "FBF-10" : 0,
            "FBF-20" : 0,
            "FBF-30" : 0,
            "GFBF-30 " : 0,
            "PGFBF-30 " : 0,
            "PFBF-30 " : 0,
            "FBPI-20 ": 0  
        }

        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"    #location of airpocket csv's 
        fileList = os.listdir(folder)                   

        for file in fileList:                                       #iterating over all files, reading in the correpsonding csv which holds the airpocket data 
            data = pd.read_csv(folder+file)             
            stat = list(data[attribute])                            #getting the attribute column in the csv and casting to list, ie all the luminance/hue etc values

            try:
                dictX[file.split("(")[0]].extend(stat)
                dictY[file.split("(")[0]] += 1
            except:
                dictX[file[0:len(file)-5]].extend(stat)
                dictY[file[0:len(file)-5]] += 1


        index=0
        graph=0

        #going to loop through the bread types and plot the distribution of each. Index, graph system needed if want to seperate into multiple graphs.
        while index-graph <= len(list(dictX.keys())):

            if index-graph == len(list(dictX.keys())):  #accounting for the condition where we hit the end of the bread types but still have an unsaved plot
                plt.title(attribute + " avg count")
                plt.legend(loc='best')
                plt.savefig("dataAnalysis/" + attribute + "AvgCount(" + str(graph+1) + ").JPG", dpi=300, format='jpg')
                plt.close()


            elif not (index+1)%12 == 0:
                key = list(dictX.keys())[index-graph]

                data = dictX[key]

                x,y = countDuplicates(data)

                y[:] = [y/dictY[key] for y in y] #dividing each count by the number of files for that bread type to get average

                #need to sort the attribute values from lowest to highest so that we can plot as a line graph, but without losing the corresponding percent area. So we need to use a dictionary 
                attributeCountDict = dict(zip(x, y))    #zipping the two value sets, attribute and area together and putting them into a dictionary, with keys = attributes and values = %areas
                attributeKeySet = list(attributeCountDict.keys())        #getting the key set (attribute)
                attributeKeySet.sort()                                  #sorting the keys (attribute) from smallest to largest
                attributeCountDict = {i: attributeCountDict[i] for i in attributeKeySet}  #now sorting the dictionary by key set

                #getting x and y data sets
                x = list(attributeCountDict.keys())
                y = list(attributeCountDict.values())


                #need to decrease sampling rate because otherwise impossible to see trends in graph, so smooth out a bit by quartering the data points with each being the average of adjacent four
                xSmooth = []
                ySmooth = []
                i = 0
                for j in range(0,math.floor(len(x)/4)):
                    try:
                        xSmooth.append((float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3]))/4)
                        ySmooth.append((float(y[i]) + float(y[i+1]) + float(y[i+2]) + float(y[i+3]))/4)
                        i+=4
                    except:
                        print(x[i])
                        print(x[i+1])
                        print(x[i+2])
                        print(x[i+3])
                        continue

                if key == "FBF-30":   
                    plt.plot(xSmooth,ySmooth, label="FBF-300s")
                elif key == "FBF-20":
                    plt.plot(xSmooth,ySmooth, label="FBF-200s")
                elif key == "FBF-10":
                    plt.plot(xSmooth,ySmooth, label="FBF-100s")
                else:      
                    plt.plot(xSmooth, ySmooth, label = key)   

            else:
                plt.title(attribute + " avg count")
                plt.legend(loc='best')
                plt.savefig("dataAnalysis/" + attribute + "AvgCount(" + str(graph+1) + ").JPG", dpi=300, format='jpg')
                plt.close()
                graph += 1
            
            index += 1




def avgPercentCount():
    """
    method to plot a line graph which shows the percentage of airpockets (number of airpockets divided by the total number of airpockets in that slice of bread)
    for each attribute value (ie score, luminence, etc), for each bread type (averaged over the images for that bread type)

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"]   

    for attribute in attributes:                #going to need a graph for each attribute, so iterate over each and construct graph for all bread types
        dictX = {      
            "FBF40 ": [],                               #initialising dictionary with bread types as keys and empty arrays which will hold the data for the attributes as values
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBPI-20 " : [] 
        }


        dictY = {                                    #initialising dictionary with bread types as keys and empty arrays which will hold the relative count for each attribute value
            "FBF5": [],                              #              X               Y                   Z                       
            "FBF25" : [],                            #  "FBF5"      -50             0.056               10                
            "FBF50" : [],                            #              -40             0.091               ...
            "FBPI5" : [],                            #              -22             0.030                
            "FBPI10" : [],                           #               ...                              
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 " : []   
        }

        dictZ = {                                    #initialising dictionary with bread types as keys and empty arrays which will hold the number of image files for each bread type
            "FBF5": 0,
            "FBF25" : 0,
            "FBF50" : 0,
            "FBPI5" : 0,
            "FBPI10" : 0,
            "LF25" : 0,
            "PF5" : 0,
            "PF25" : 0,
            "PF50" : 0,
            "PPI5" : 0,
            "WF" : 0,
            "WF-100 " : 0,
            "Control" : 0,
            "FBF-30 " : 0,
            "FBF-10" : 0,
            "FBF-20" : 0,
            "FBF-30" : 0,
            "GFBF-30 " : 0,
            "PGFBF-30 " : 0,
            "PFBF-30 " : 0,
            "FBF40 ": 0,
            "FBPI-20 ": 0   
        }

        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"    #location of airpocket csv's 
        fileList = os.listdir(folder)                   

        for file in fileList:                                       #iterating over all files, reading in the correpsonding csv which holds the airpocket data 
            data = pd.read_csv(folder+file)             
            stat = list(data[attribute])                            

            values, counts = countDuplicates(stat)          #returns two arrays, an array of values and an array of counts showing the number of times that value was found in the stat array

            counts[:] = [y/len(stat) for y in counts]       #dividing by the length of the data file, which should be the total number of airpockets

            for i in range(0, len(values)):
                """
                if not values[i] in dictX[file.split("(")[0]]:      #if value doesn't already exist
                    dictX[file.split("(")[0]].append(values[i])    
                    dictY[file.split("(")[0]].append(counts[i])
                else:
                    index = dictX[file.split("(")[0]].index(values[i])      #finding the index where the value already exists
                    dictY[file.split("(")[0]][index] += counts[i]           #adding the count to that index

                dictZ[file.split("(")[0]] += 1                  #holds the number of images for each bread type
                """

                try:
                    cond = not values[i] in dictX[file.split("(")[0]]
                except:
                    cond = not values[i] in dictX[file[0:len(file)-5]]

                if cond:      #if value doesn't already exist
                    try:
                        dictX[file.split("(")[0]].append(values[i])
                        dictY[file.split("(")[0]].append(counts[i])
                    except:
                        dictX[file[0:len(file)-5]].append(values[i])
                        dictY[file[0:len(file)-5]].append(counts[i])

                else:
                    try:
                        index = dictX[file.split("(")[0]].index(values[i])      #finding the index where the value already exists
                        dictY[file.split("(")[0]][index] += counts[i]           #adding the count to that index
                    except:
                        index = dictX[file[0:len(file)-5]].index(values[i])
                        dictY[file[0:len(file)-5]][index] += counts[i]

            try:
                dictZ[file.split("(")[0]] += 1          #holds the number of images for each bread type
            except:
                dictZ[file[0:len(file)-5]] += 1

            
        index = 0
        graph = 0

        #going to loop through the bread types and plot the distribution of each. Index, graph system needed if want to seperate into multiple graphs.
        while index-graph <= len(list(dictX.keys())):

            if index - graph == len(list(dictX.keys())):  #accounting for the condition where we hit the end of the bread types but still have an unsaved plot
                plt.title(attribute + " avg percent count")
                plt.legend(loc='best')
                plt.savefig("dataAnalysis/" + attribute + "Avg%Count(" + str(graph+1) + ").JPG", dpi=300, format='jpg')
                plt.close()
                            
            
            elif not (index+1)%12 == 0:
                xKey = list(dictX.keys())[index-graph]          #getting the key, aka bread type. Note that xKey should equal yKey always, so redundant to have two variables
                yKey = list(dictY.keys())[index-graph]

                x = dictX[xKey]
                y = dictY[yKey]

                y[:] = [y/dictZ[yKey] for y in y]                #dividing by the number of images for the bread type to average

                #need to sort the attribute values from lowest to highest so that we can plot as a line graph, but without losing the corresponding percent area. So we need to use a dictionary 
                attributeCountDict = dict(zip(x, y))    #zipping the two value sets, attribute and area together and putting them into a dictionary, with keys = attributes and values = %areas
                attributeKeySet = list(attributeCountDict.keys())        #getting the key set (attribute)
                attributeKeySet.sort()                                  #sorting the keys (attribute) from smallest to largest
                attributeCountDict = {i: attributeCountDict[i] for i in attributeKeySet}  #now sorting the dictionary by key set

                #getting x and y data sets
                x = list(attributeCountDict.keys())
                y = list(attributeCountDict.values())

                #need to decrease sampling rate because otherwise impossible to see trends in graph, so smooth out a bit by quartering the data points with each being the average of adjacent four
                xSmooth = []
                ySmooth = []
                i = 0
                for j in range(0,math.floor(len(x)/4)):
                    try:
                        xSmooth.append((float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3]))/4)
                        ySmooth.append((float(y[i]) + float(y[i+1]) + float(y[i+2]) + float(y[i+3]))/4)
                        i+=4
                    except:
                        continue

                if xKey == "FBF-30":   
                    plt.plot(xSmooth,ySmooth, label="FBF-300s")
                elif xKey == "FBF-20":
                    plt.plot(xSmooth,ySmooth, label="FBF-200s")
                elif xKey == "FBF-10":
                    plt.plot(xSmooth,ySmooth, label="FBF-100s")
                else:      
                    plt.plot(xSmooth, ySmooth, label = xKey)   



            else:                           #the case where index +1 is a multiple of ...
                plt.title(attribute + " avg percent count")
                plt.legend(loc='best')
                plt.savefig("dataAnalysis/" + attribute + "Avg%Count(" + str(graph+1) + ").JPG", dpi=300, format='jpg')
                plt.close()

                graph += 1
            
            index += 1
            


def percentArea():
    """
    method to plot a linegraph showing the percentage of total bread area that the airpockets for each attribute value makes up, ie what percentage of the bread area has airpockets of luminance = 50? 
    for each breadtype.

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"]

    for attribute in attributes:
        dictX = {               #these are going to the attribute values
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
        }

        dictY = {               #these are going to be the percent area, in 1 to 1 correspondence with the attribute values array,
            "FBF5": [],                     # eg: dictX:    key     luminence       dictY:  key     %area
            "FBF25" : [],                   #               "FBF5"   7                      "FBF5"  0.04
            "FBF50" : [],                   #                        13                             0.093
            "FBPI5" : [],                   #                        27                             0.11
            "FBPI10" : [],                  #                        14                             0.2
            "LF25" : [],                    #               "FBF25"  8                      "FBF25" 0.01
            "PF5" : [],                     #                        35                             0.27    
            "PF25" : [],                    #                        ...
            "PF50" : [],                    #
            "PPI5" : [],                    #   eventually will stitch the value set of dictX (the attributes, in the above case luminance), with the value set of dictY, %area
            "WF" : [],                       #   and sort by luminence so that we can do a standard line plot with attribute on the x axis and %area on the y
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
       
        }

        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"
        fileList = os.listdir(folder)

        for file in fileList: 

            #need to get the total area of the interior of the slice of bread (without crust)
            img, scale = imgSearch(file.split('.')[0])
            canvasName = "crustCanvas/" + file.split('.')[0] + ".JPG"
            canvas = cv2.imread(canvasName)

            """
            if canvas is None:
                canvasName = "crustCanvas/" + file[0:len(file)-7] + file[len(file)-6:len(file)-5] + ".JPG" 
                canvas = cv2.imread(canvasName)

            if canvas is None:
                canvasName = "crustCanvas/" + file[0:len(file)-7] + " " + file[len(file)-7:len(file)-4] + ".JPG"
                canvas = cv2.imread(canvasName)
            """
            breadInterior = removeCrust(img, canvas)      

            # thresholding with very high thresh to segment the image into the bread and the white background, finding contours on this segmentation, 
            # with the biggest being the external contour that contains the whole bread slice
            grey = cv2.cvtColor(breadInterior, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)[1]
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  
            contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

            
            breadArea = cv2.contourArea(contoursSorted[len(contoursSorted)-1])      #area of the biggest contour
            breadArea = breadArea/math.pow(scale,2)                                 #applying scale to shift from pixels to mm2

            data = pd.read_csv(folder+file)

            attributeData = list(data[attribute])     #this is the list of whatever attribute, ie luminence, for each airpocket
            areaData = list(data["area"])             #this is the list of areas (for each airpocket)

            for i in range(0, len(attributeData)-1):
                """
                if not attributeData[i] in dictX.get(file.split("(")[0]):    #checking if the luminence value is already in the list for that bread type (contained in dictX)
                    dictX.get(file.split("(")[0]).append(attributeData[i])      #adding luminence value to list if it's not already there
                    dictY[file.split("(")[0]].append(areaData[i]/breadArea)     #adding the corresponding percent area to the percent area set for that bread type in dictY
    
                else:
                    index = dictX.get(file.split("(")[0]).index(attributeData[i])       #finding position in dictX's values array where the attribute value already exists
                    dictY[file.split("(")[0]][index] += areaData[i]/breadArea           #adding the percent area to the value corresponding to that index position in the dictY area array
                """

                try:
                    cond = not attributeData[i] in dictX[file.split("(")[0]]
                except:
                    cond = not attributeData[i] in dictX[file[0:len(file)-5]]

                if cond:      #if value doesn't already exist
                    try:
                        dictX[file.split("(")[0]].append(attributeData[i])
                        dictY[file.split("(")[0]].append(areaData[i]/breadArea)
                    except:
                        dictX[file[0:len(file)-5]].append(attributeData[i])
                        dictY[file[0:len(file)-5]].append(areaData[i]/breadArea)

                else:
                    try:
                        index = dictX[file.split("(")[0]].index(attributeData[i])      #finding the index where the value already exists
                        dictY[file.split("(")[0]][index] += areaData[i]/breadArea          #adding the count to that index
                    except:
                        index = dictX[file[0:len(file)-5]].index(attributeData[i])
                        dictY[file[0:len(file)-5]][index] += areaData[i]/breadArea
 

        #end of file loop
        index = 0
        graph = 0

        #going to loop through the bread types and plot the distribution of each. Index graph system needed if want to seperate into multiple graphs.
        while(index - graph <= len(dictX.keys())):

            if index-graph == len(dictX.keys()):    #accounting for the condition where we hit the end of the bread types but still have an unsaved plot
                plt.title(attribute + " percent area")
                plt.legend(loc='best')
                plt.savefig("dataAnalysis/" + attribute + "%Area(" + str(graph+1) + ").JPG", dpi=300, format='jpg')
                plt.close()

            elif not (index+1)%12 == 0:
                xKey = list(dictX.keys())[index-graph]          #getting the key, aka bread type. Note that xKey should equal yKey always, so redundant to have two variables
                yKey = list(dictY.keys())[index-graph]


                #need to sort the attribute values from lowest to highest so that we can plot as a line graph, but without losing the corresponding percent area. So we need to use a dictionary 
                attributeAreaDict = dict(zip(dictX[xKey],dictY[yKey]))    #zipping the two value sets, attribute and area together and putting them into a dictionary, with keys = attributes and values = %areas
                attributeKeySet = list(attributeAreaDict.keys())        #getting the key set (attribute)
                attributeKeySet.sort()                                  #sorting the keys (attribute) from smallest to largest
                attributeAreaDict = {i: attributeAreaDict[i] for i in attributeKeySet}  #now sorting the dictionary by key set
                #getting x and y data sets
                x = list(attributeAreaDict.keys())
                y = list(attributeAreaDict.values())

                #need to decrease sampling rate because otherwise impossible to see trends in graph, so smooth out a bit by quartering the data points with each being the average of adjacent four
                xSmooth = []
                ySmooth = []
                i = 0
                for j in range(0,math.floor(len(x)/4)):
                        xSmooth.append((float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3]))/4)
                        ySmooth.append((float(y[i]) + float(y[i+1]) + float(y[i+2]) + float(y[i+3]))/4)
                        i+=4

                if xKey == "FBF-30":   
                    plt.plot(xSmooth,ySmooth, label="FBF-300s")
                elif xKey == "FBF-20":
                    plt.plot(xSmooth,ySmooth, label="FBF-200s")
                elif xKey == "FBF-10":
                    plt.plot(xSmooth,ySmooth, label="FBF-100s")
                else:      
                    plt.plot(xSmooth, ySmooth, label = xKey)   

            else:                                       #the case where index +1 is a multiple of 4
                plt.title(attribute + " percent area")
                plt.legend(loc='best')
                plt.savefig("dataAnalysis/" + attribute + "%Area(" + str(graph+1) + ").JPG", dpi=300, format='jpg')
                plt.close()
                graph += 1
            
            index += 1




def avgPercentArea():
    """
    method to plot a linegraph showing the percentage of total bread area that the airpockets for each attribute value makes up, ie what percentage of the bread area has airpockets of luminance = 50? 
    for each breadtype averaged over all the images of that bread.

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"]

    for attribute in attributes:
        dictX = {               #these are going to the attribute values
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
        }

        dictY = {               #these are going to be the percent area, in 1 to 1 correspondence with the attribute values array,
            "FBF5": [],                     # eg: dictX:    key     luminence       dictY:  key     %area       dictZ:   key       number
            "FBF25" : [],                   #               "FBF5"   7                      "FBF5"  0.04                 "FBF5"    10
            "FBF50" : [],                   #                        13                             0.093                "FBF25"   10
            "FBPI5" : [],                   #                        27                             0.11
            "FBPI10" : [],                  #                        14                             0.2
            "LF25" : [],                    #               "FBF25"  8                      "FBF25" 0.01
            "PF5" : [],                     #                        35                             0.27    
            "PF25" : [],                    #                        ...
            "PF50" : [],                    #
            "PPI5" : [],                    #   eventually will stitch the value set of dictX (the attributes, in the above case luminance), with the value set of dictY, %area
            "WF" : [],                       #   and sort by luminence so that we can do a standard line plot with attribute on the x axis and %area on the y
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []                
        }

        dictZ = {               #initialising dictionary with bread types as keys and empty arrays which will hold the number of image files for each bread type
            "FBF5": 0,
            "FBF25" : 0,
            "FBF50" : 0,
            "FBPI5" : 0,
            "FBPI10" : 0,
            "LF25" : 0,
            "PF5" : 0,
            "PF25" : 0,
            "PF50" : 0,
            "PPI5" : 0,
            "WF" : 0,
            "WF-100 " : 0,
            "Control" : 0,
            "FBF-30 " : 0,
            "FBF-10" : 0,
            "FBF-20" : 0,
            "FBF-30" : 0,
            "GFBF-30 " : 0,
            "PGFBF-30 " : 0,
            "PFBF-30 " : 0,
            "FBF40 ": 0,
            "FBPI-20 ": 0  
        }

        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"
        fileList = os.listdir(folder)

        for file in fileList:

            #need to get the total area of the interior of the slice of bread (without crust)
            img, scale = imgSearch(file.split('.')[0])
            canvasName = "crustCanvas/" + file.split('.')[0] + ".JPG"
            print(canvasName)
            canvas = cv2.imread(canvasName)

            """
            if canvas is None:
                canvasName = "crustCanvas/" + file[0:len(file)-7] + file[len(file)-6:len(file)-5] + ".JPG" 
                print(canvasName)
                canvas = cv2.imread(canvasName)

            if canvas is None:
                canvasName = "crustCanvas/" + file[0:len(file)-7] + " " + file[len(file)-7:len(file)-4] + ".JPG"
                print(canvasName)
                canvas = cv2.imread(canvasName)
            """
            breadInterior = removeCrust(img, canvas)      

            # thresholding with very high thresh to segment the image into the bread and the white background, finding contours on this segmentation, 
            # with the biggest being the external contour that contains the whole bread slice
            grey = cv2.cvtColor(breadInterior, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)[1]
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  
            contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

            
            breadArea = cv2.contourArea(contoursSorted[len(contoursSorted)-1])      #area of the biggest contour
            breadArea = breadArea/math.pow(scale,2)                                 #applying scale to shift from pixels to mm2

            data = pd.read_csv(folder+file)

            attributeData = list(data[attribute])     #this is the list of whatever attribute, ie luminence, for each airpocket
            areaData = list(data["area"])             #this is the list of areas (for each airpocket)

            for i in range(0, len(attributeData)-1):
                """
                if not attributeData[i] in dictX.get(file.split("(")[0]):    #checking if the luminence value is already in the list for that bread type (contained in dictX)
                    dictX.get(file.split("(")[0]).append(attributeData[i])      #adding luminence value to list if it's not already there
                    dictY[file.split("(")[0]].append(areaData[i]/breadArea)     #adding the corresponding percent area to the percent area set for that bread type in dictY
    
                else:
                    index = dictX.get(file.split("(")[0]).index(attributeData[i])       #finding position in dictX's values array where the attribute value already exists
                    dictY[file.split("(")[0]][index] += areaData[i]/breadArea           #adding the percent area to the value corresponding to that index position in the dictY area array
                """


                try:
                    cond = not attributeData[i] in dictX[file.split("(")[0]]
                except:
                    cond = not attributeData[i] in dictX[file[0:len(file)-5]]

                if cond:      #if value doesn't already exist
                    try:
                        dictX[file.split("(")[0]].append(attributeData[i])
                        dictY[file.split("(")[0]].append(areaData[i]/breadArea)
                    except:
                        dictX[file[0:len(file)-5]].append(attributeData[i])
                        dictY[file[0:len(file)-5]].append(areaData[i]/breadArea)

                else:
                    try:
                        index = dictX[file.split("(")[0]].index(attributeData[i])      #finding the index where the value already exists
                        dictY[file.split("(")[0]][index] += areaData[i]/breadArea          #adding the count to that index
                    except:
                        index = dictX[file[0:len(file)-5]].index(attributeData[i])
                        dictY[file[0:len(file)-5]][index] += areaData[i]/breadArea
 
            try:
                dictZ[file.split("(")[0]] += 1          #holds the number of images for each bread type
            except:
                dictZ[file[0:len(file)-5]] += 1   
       


        #end of file loop
        index = 0
        graph = 0

        #now going to loop through they bread types, and plot the distribution of each. Need index and graph features if want to plot on seperate graphs, acts like a for loop.
        while(index - graph <= len(dictX.keys())):

            if index-graph == len(dictX.keys()):    #accounting for the condition where we hit the end of the bread types but still have an unsaved plot
                plt.title(attribute + " avg percent area")
                plt.legend(loc='best')
                plt.savefig("dataAnalysis/" + attribute + "Avg%Area(" + str(graph+1) + ").JPG", dpi=300, format='jpg')
                plt.close()

            elif not (index+1)%12 == 0:
                xKey = list(dictX.keys())[index-graph]
                yKey = list(dictY.keys())[index-graph]


                #need to sort the attribute values from lowest to highest so that we can plot as a line graph, but without losing the corresponding percent area. So we need to use a dictionary 
                attributeAreaDict = dict(zip(dictX[xKey],dictY[yKey]))    #zipping the two value sets, attribute and area together and putting them into a dictionary, with keys = attributes and values = %areas
                attributeKeySet = list(attributeAreaDict.keys())        #getting the key set (attribute)
                attributeKeySet.sort()                                  #sorting the keys (attribute) from smallest to largest
                attributeAreaDict = {i: attributeAreaDict[i] for i in attributeKeySet}  #now sorting the dictionary by key set
                #getting x and y data sets
                x = list(attributeAreaDict.keys())
                y = list(attributeAreaDict.values())

                y[:] = [y/dictZ[yKey] for y in y]           #dividing each area value by the number of image files for that bread type

                #need to decrease sampling rate because otherwise impossible to see trends in graph, so smooth out a bit by quartering the data points with each being the average of adjacent four
                xSmooth = []
                ySmooth = []
                i = 0
                for j in range(0,math.floor(len(x)/4)):
                    try:
                        xSmooth.append((float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3]))/4)
                        ySmooth.append((float(y[i]) + float(y[i+1]) + float(y[i+2]) + float(y[i+3]))/4)
                        i+=4
                    except:
                        continue
                
                if xKey == "FBF-30":   
                    plt.plot(xSmooth,ySmooth, label="FBF-300s")
                elif xKey == "FBF-20":
                    plt.plot(xSmooth,ySmooth, label="FBF-200s")
                elif xKey == "FBF-10":
                    plt.plot(xSmooth,ySmooth, label="FBF-100s")
                else:      
                    plt.plot(xSmooth, ySmooth, label = xKey)  

            else:                                       #the case where index +1 is a multiple of 4
                plt.title(attribute + " avg percent area")
                plt.legend(loc='best')
                plt.savefig("dataAnalysis/" + attribute + "Avg%Area(" + str(graph+1) + ").JPG", dpi=300, format='jpg')
                plt.close()
                graph += 1
            
            index += 1



def smallestToBiggest():
    """
    method to print the average mean, min, max, range, and std of each attribute (score, luminance, hue, saturation, value) for each bread type.
    will print to a csv, where they are ordered from smallest mean to greatest.

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"] 

    for attribute in attributes:

        dictX = {               
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
        }

        dictAvgs = {
            "FBF5": 0,
            "FBF25" : 0,
            "FBF50" : 0,
            "FBPI5" : 0,
            "FBPI10" : 0,
            "LF25" : 0,
            "PF5" : 0,
            "PF25" : 0,
            "PF50" : 0,
            "PPI5" : 0,
            "WF" : 0, 
            "WF-100 " : 0,
            "Control" : 0,
            "FBF-30 " : 0,
            "FBF-10" : 0,
            "FBF-20" : 0,
            "FBF-30" : 0,
            "GFBF-30 " : 0,
            "PGFBF-30 " : 0,
            "PFBF-30 " : 0,
            "FBF40 ": 0,
            "FBPI-20 ": 0           
        }


        dictFinal = {
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
        }


        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"    #location of airpocket csv's 
        fileList = os.listdir(folder)                   

        for file in fileList:                    #iterating over all files, reading in the correpsonding csv which holds the airpocket data 
            data = pd.read_csv(folder+file)             
            stat = list(data[attribute])  

            data = [np.average(stat), np.min(stat), np.max(stat), np.max(stat)-np.min(stat), np.std(stat)]

            try:
                dictX[file.split("(")[0]].extend(data)
            except:
                dictX[file[0:len(file)-5]].extend(data) 

        
        #writing title line in csv
        with open("outputData/lineup(" + attribute + ").csv", 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["bread", "mean", "min", "max", "range", "std"])
        

        #now will iterate over bread types
        for key in dictX.keys():

            means = []
            mins = []
            maxes = []
            range = []
            std = []

            index = 0

            while index <= len(dictX[key])-5:   #iterating over the length of the dictionary for that bread type in steps of 5. Adding each mean, min, etc to an array.
                means.append(dictX[key][index])
                mins.append(dictX[key][index + 1])
                maxes.append(dictX[key][index + 2])
                range.append(dictX[key][index + 3])
                std.append(dictX[key][index + 4])
                index += 5


            dictAvgs[key] = np.average(means)
            dictFinal[key] = [np.average(means), np.average(mins), np.average(maxes), np.average(range), np.average(std)]
        

        dictAvgs = dict(sorted(dictAvgs.items(), key = lambda x:x[1]))  #sorting the dictionary of averages to get order

        for key in dictAvgs.keys(): #now iterating over bread types, in the order of dictAvgs

            data = []
            
            if key == "FBF-30":   
                label = "FBF-300s"
            elif key == "FBF-20":
                label = "FBF-200s"
            elif key == "FBF-10":
                label = "FBF-100s"
            else:      
                label = key


            data.append(label)    
            data.extend(dictFinal[key]) #getting the condensed/final data for the bread type from dictFinal

            #writing to csv
            with open("outputData/lineup(" + attribute + ").csv", 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)




def avgCountBoxplot():
    """
    method to plot a box plot which shows how many airpockets there are for each attribute value (ie score, luminence, etc), for each bread type (averaged over the images for that bread type)

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"]   #going to need a graph for each attribute, so iterate over each and construct graph for all bread types

    for attribute in attributes:                #going to need a graph for each attribute, so iterate over each and construct graph for all bread types
        dictX = {                                    #initialising dictionary with bread types as keys and empty arrays which will hold the data for the attributes as values
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
        }

        dictY = {                                    #initialising dictionary with bread types as keys and empty arrays which will hold the number of image files for each bread type
            "FBF5": 0,
            "FBF25" : 0,
            "FBF50" : 0,
            "FBPI5" : 0,
            "FBPI10" : 0,
            "LF25" : 0,
            "PF5" : 0,
            "PF25" : 0,
            "PF50" : 0,
            "PPI5" : 0,
            "WF" : 0,
            "WF-100 " : 0,
            "Control" : 0,
            "FBF-30 " : 0,
            "FBF-10" : 0,
            "FBF-20" : 0,
            "FBF-30" : 0,
            "GFBF-30 " : 0,
            "PGFBF-30 " : 0,
            "PFBF-30 " : 0,
            "FBF40 ": 0,
            "FBPI-20 ": 0   
        }

        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"    #location of airpocket csv's 
        fileList = os.listdir(folder)                   

        for file in fileList:                                       #iterating over all files, reading in the correpsonding csv which holds the airpocket data 
            data = pd.read_csv(folder+file)             
            stat = list(data[attribute])                            #getting the attribute column in the csv and casting to list, ie all the luminance/hue etc values

            try:
                dictX[file.split("(")[0]].extend(stat)
                dictY[file.split("(")[0]] += 1
            except:
                dictX[file[0:len(file)-5]].extend(stat)
                dictY[file[0:len(file)-5]] += 1


        boxes = []
    
        #going to loop through the bread types and plot a boxplot for each.
        for key in dictX.keys():
            data = dictX[key]
            x,y = countDuplicates(data)
            y[:] = [y/dictY[key] for y in y] #dividing each count by the number of files for that bread type to get average

            box = []
            #boxplot method takes an array and plots the distribution of values within that array, so we need to get values from value, count format back to repeated values.
            #so I'm just going to add the value to a new array however many times the count for that value is. Should preserve the distribution that way
            #need for loop now to iterate over the x and y, and construct a new array which can be made into a boxplot
            for i in range(0,len(y)-1):         
                arr = np.full(math.floor(y[i]), x[i], dtype="double")
                box.extend(arr)
            
            boxes.append(box)


        labels = []

        for key in dictX.keys():               
            if key == "FBF-30":   
                labels.append("FBF-300s")
            elif key == "FBF-20":
                labels.append("FBF-200s")
            elif key == "FBF-10":
                labels.append("FBF-100s")
            else:      
                labels.append(key)

        fig = plt.figure(figsize = (20,7))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(boxes)
        ax.set_xticklabels(labels)
        plt.title(attribute + " avg count")
        plt.savefig("dataAnalysis/boxplot/" + attribute + "AvgCount.JPG", dpi=300, format='jpg')
        plt.close()





def avgPercentCountBoxplot():
    """
    method to plot a box plot which shows the percentage of airpockets (number of airpockets divided by the total number of airpockets in that slice of bread)
    for each attribute value (ie score, luminence, etc), for each bread type (averaged over the images for that bread type)

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"]   

    for attribute in attributes:                #going to need a graph for each attribute, so iterate over each and construct graph for all bread types
        dictX = {                                    #initialising dictionary with bread types as keys and empty arrays which will hold the data for the attributes as values
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
        }


        dictY = {                                    #initialising dictionary with bread types as keys and empty arrays which will hold the relative count for each attribute value
            "FBF5": [],                              #              X               Y                   Z                       
            "FBF25" : [],                            #  "FBF5"      -50             0.056               10                
            "FBF50" : [],                            #              -40             0.091               ...
            "FBPI5" : [],                            #              -22             0.030                
            "FBPI10" : [],                           #               ...                              
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   


        }

        dictZ = {                                    #initialising dictionary with bread types as keys and empty arrays which will hold the number of image files for each bread type
            "FBF5": 0,
            "FBF25" : 0,
            "FBF50" : 0,
            "FBPI5" : 0,
            "FBPI10" : 0,
            "LF25" : 0,
            "PF5" : 0,
            "PF25" : 0,
            "PF50" : 0,
            "PPI5" : 0,
            "WF" : 0,
            "WF-100 " : 0,
            "Control" : 0,
            "FBF-30 " : 0,
            "FBF-10" : 0,
            "FBF-20" : 0,
            "FBF-30" : 0,
            "GFBF-30 " : 0,
            "PGFBF-30 " : 0,
            "PFBF-30 " : 0,
            "FBF40 ": 0,
            "FBPI-20 ": 0   
        }

        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"    #location of airpocket csv's 
        fileList = os.listdir(folder)                   

        for file in fileList:                                       #iterating over all files, reading in the correpsonding csv which holds the airpocket data 
            data = pd.read_csv(folder+file)             
            stat = list(data[attribute])                            

            values, counts = countDuplicates(stat)          #returns two arrays, an array of values and an array of counts showing the number of times that value was found in the stat array

            counts[:] = [y/len(stat) for y in counts]       #dividing by the length of the data file, which should be the total number of airpockets

            for i in range(0, len(values)):
                try:
                    cond = not values[i] in dictX[file.split("(")[0]]
                except:
                    cond = not values[i] in dictX[file[0:len(file)-5]]

                if cond:      #if value doesn't already exist
                    try:
                        dictX[file.split("(")[0]].append(values[i])
                        dictY[file.split("(")[0]].append(counts[i])
                    except:
                        dictX[file[0:len(file)-5]].append(values[i])
                        dictY[file[0:len(file)-5]].append(counts[i])

                else:
                    try:
                        index = dictX[file.split("(")[0]].index(values[i])      #finding the index where the value already exists
                        dictY[file.split("(")[0]][index] += counts[i]           #adding the count to that index
                    except:
                        index = dictX[file[0:len(file)-5]].index(values[i])
                        dictY[file[0:len(file)-5]][index] += counts[i]

            try:
                dictZ[file.split("(")[0]] += 1          #holds the number of images for each bread type
            except:
                dictZ[file[0:len(file)-5]] += 1

            
        boxes = []
        labels = []
        #going to loop through the bread types and plot the boxplot of each. 
        for key in dictX.keys():                          
            
            x = dictX[key]
            y = dictY[key]
            y[:] = [y/dictZ[key] for y in y]                #dividing by the number of images for the bread type to average
            
            box = []
            #boxplot method takes an array and plots the distribution of values within that array, so we need to get values from value, count format back to repeated values.
            #so I'm just going to add the value to a new array however many times the percent count, multiplied by 1 thousand because we need to round (ie 1%/0.01 becomes 10), for that value is. Should preserve the distribution that way
            #need for loop now to iterate over the x and y, and construct a new array which can be made into a boxplot
            for i in range(0,len(y)-1):        
                arr = np.full(math.floor(y[i]*1000), x[i], dtype="double")
                box.extend(arr)
            
            boxes.append(box)

            if key == "FBF-30":   
                labels.append("FBF-300s")
            elif key == "FBF-20":
                labels.append("FBF-200s")
            elif key == "FBF-10":
                labels.append("FBF-100s")
            else:      
                labels.append(key)


        fig = plt.figure(figsize = (20,7))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(boxes)
        ax.set_xticklabels(labels)
        plt.title(attribute + " avg percent count")
        plt.savefig("dataAnalysis/boxplot/" + attribute + "Avg%Count.JPG", dpi=300, format='jpg')
        plt.close()



def percentAreaBoxplot():
    """
    method to plot a boxplot showing the percentage of total bread area that the airpockets for each attribute value makes up, ie what percentage of the bread area has airpockets of luminance = 50? 
    for each breadtype.

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"]

    for attribute in attributes:
        print(attribute)
        dictX = {               #these are going to the attribute values
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
            }

        dictY = {               #these are going to be the percent area, in 1 to 1 correspondence with the attribute values array,
            "FBF5": [],                     # eg: dictX:    key     luminence       dictY:  key     %area
            "FBF25" : [],                   #               "FBF5"   7                      "FBF5"  0.04
            "FBF50" : [],                   #                        13                             0.093
            "FBPI5" : [],                   #                        27                             0.11
            "FBPI10" : [],                  #                        14                             0.2
            "LF25" : [],                    #               "FBF25"  8                      "FBF25" 0.01
            "PF5" : [],                     #                        35                             0.27    
            "PF25" : [],                    #                        ...
            "PF50" : [],                    #
            "PPI5" : [],                    #   eventually will stitch the value set of dictX (the attributes, in the above case luminance), with the value set of dictY, %area
            "WF" : [],                       #   and sort by luminence so that we can do a standard line plot with attribute on the x axis and %area on the y
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": []   
            }

        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"
        fileList = os.listdir(folder)

        for file in fileList:
            #need to get the total area of the interior of the slice of bread (without crust)
            img, scale = imgSearch(file.split('.')[0])
            canvasName = "crustCanvas/" + file.split('.')[0] + ".JPG"
            print(canvasName)
            canvas = cv2.imread(canvasName)

            """
            if canvas is None:
                canvasName = "crustCanvas/" + file[0:len(file)-7] + file[len(file)-6:len(file)-5] + ".JPG" 
                print(canvasName)
                canvas = cv2.imread(canvasName)

            if canvas is None:
                canvasName = "crustCanvas/" + file[0:len(file)-7] + " " + file[len(file)-7:len(file)-4] + ".JPG"
                print(canvasName)
                canvas = cv2.imread(canvasName)
            """
            breadInterior = removeCrust(img, canvas)       

            # thresholding with very high thresh to segment the image into the bread and the white background, finding contours on this segmentation, 
            # with the biggest being the external contour that contains the whole bread slice
            grey = cv2.cvtColor(breadInterior, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)[1]
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  
            contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

            
            breadArea = cv2.contourArea(contoursSorted[len(contoursSorted)-1])      #area of the biggest contour
            breadArea = breadArea/math.pow(scale,2)                                 #applying scale to shift from pixels to mm2

            data = pd.read_csv(folder+file)

            attributeData = list(data[attribute])     #this is the list of whatever attribute, ie luminence, for each airpocket
            areaData = list(data["area"])             #this is the list of areas (for each airpocket)

            for i in range(0, len(attributeData)-1):
                #if not attributeData[i] in dictX.get(file.split("(")[0]):    #checking if the luminence value is already in the list for that bread type (contained in dictX)
                #    dictX.get(file.split("(")[0]).append(attributeData[i])      #adding luminence value to list if it's not already there
                #    dictY[file.split("(")[0]].append(areaData[i]/breadArea)     #adding the corresponding percent area to the percent area set for that bread type in dictY
    
                #else:
                #    index = dictX.get(file.split("(")[0]).index(attributeData[i])       #finding position in dictX's values array where the attribute value already exists
                #    dictY[file.split("(")[0]][index] += areaData[i]/breadArea           #adding the percent area to the value corresponding to that index position in the dictY area array



                try:
                    cond = not attributeData[i] in dictX[file.split("(")[0]]
                except:
                    cond = not attributeData[i] in dictX[file[0:len(file)-5]]

                if cond:      #if value doesn't already exist
                    try:
                        dictX[file.split("(")[0]].append(attributeData[i])
                        dictY[file.split("(")[0]].append(areaData[i]/breadArea)
                    except:
                        dictX[file[0:len(file)-5]].append(attributeData[i])
                        dictY[file[0:len(file)-5]].append(areaData[i]/breadArea)

                else:
                    try:
                        index = dictX[file.split("(")[0]].index(attributeData[i])      #finding the index where the value already exists
                        dictY[file.split("(")[0]][index] += areaData[i]/breadArea          #adding the count to that index
                    except:
                        index = dictX[file[0:len(file)-5]].index(attributeData[i])
                        dictY[file[0:len(file)-5]][index] += areaData[i]/breadArea


        boxes = []
        labels = []
        #going to loop through the bread types and plot the boxplot of each. 
        for key in dictX.keys():

            #getting x and y data sets
            x = list(dictX[key])
            y = list(dictY[key])
            
            box = []
            #boxplot method takes an array and plots the distribution of values within that array, so we need to get values from value, count format back to repeated values.
            #so I'm just going to add the value to a new array however many times the percent count, multiplied by 1 thousand because we need to round (ie 1%/0.01 becomes 10), for that value is. Should preserve the distribution that way
            #need for loop now to iterate over the x and y, and construct a new array which can be made into a boxplot
            for i in range(0,len(y)-1):         #iterating over all the attribute values
                arr = np.full(math.floor(y[i]*1000), x[i], dtype="double")
                box.extend(arr)

            boxes.append(box)

            if key == "FBF-30":   
                labels.append("FBF-300s")
            elif key == "FBF-20":
                labels.append("FBF-200s")
            elif key == "FBF-10":
                labels.append("FBF-100s")
            else:      
                labels.append(key)
            


        fig = plt.figure(figsize = (20,7))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(boxes)
        ax.set_xticklabels(labels)
        plt.title(attribute + " percent area")
        plt.savefig("dataAnalysis/boxplot/" + attribute + "%Area.JPG", dpi=300, format='jpg')
        plt.close()
 




def avgPercentAreaBoxplot():
    """
    method to plot a box plot showing the percentage of total bread area that the airpockets for each attribute value makes up, ie what percentage of the bread area has airpockets of luminance = 50? 
    for each breadtype averaged over all the images of that bread.

    """
    attributes = ["score", "luminance", "hue", "saturation", "value"]

    for attribute in attributes:
        print(attribute)
        dictX = {               #these are going to the attribute values
            "FBF5": [],
            "FBF25" : [],
            "FBF50" : [],
            "FBPI5" : [],
            "FBPI10" : [],
            "LF25" : [],
            "PF5" : [],
            "PF25" : [],
            "PF50" : [],
            "PPI5" : [],
            "WF" : [],
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": [] 
            }

        dictY = {               #these are going to be the percent area, in 1 to 1 correspondence with the attribute values array,
            "FBF5": [],                     # eg: dictX:    key     luminence       dictY:  key     %area       dictZ:   key       number
            "FBF25" : [],                   #               "FBF5"   7                      "FBF5"  0.04                 "FBF5"    10
            "FBF50" : [],                   #                        13                             0.093                "FBF25"   10
            "FBPI5" : [],                   #                        27                             0.11
            "FBPI10" : [],                  #                        14                             0.2
            "LF25" : [],                    #               "FBF25"  8                      "FBF25" 0.01
            "PF5" : [],                     #                        35                             0.27    
            "PF25" : [],                    #                        ...
            "PF50" : [],                    #
            "PPI5" : [],                    #   eventually will stitch the value set of dictX (the attributes, in the above case luminance), with the value set of dictY, %area
            "WF" : [],                       #   and sort by luminence so that we can do a standard line plot with attribute on the x axis and %area on the y
            "WF-100 " : [],
            "Control" : [],
            "FBF-30 " : [],
            "FBF-10" : [],
            "FBF-20" : [],
            "FBF-30" : [],
            "GFBF-30 " : [],
            "PGFBF-30 " : [],
            "PFBF-30 " : [],
            "FBF40 ": [],
            "FBPI-20 ": [] 
        }

        dictZ = {               #initialising dictionary with bread types as keys and empty arrays which will hold the number of image files for each bread type
            "FBF5": 0,
            "FBF25" : 0,
            "FBF50" : 0,
            "FBPI5" : 0,
            "FBPI10" : 0,
            "LF25" : 0,
            "PF5" : 0,
            "PF25" : 0,
            "PF50" : 0,
            "PPI5" : 0,
            "WF" : 0,
            "WF-100 " : 0,
            "Control" : 0,
            "FBF-30 " : 0,
            "FBF-10" : 0,
            "FBF-20" : 0,
            "FBF-30" : 0,
            "GFBF-30 " : 0,
            "PGFBF-30 " : 0,
            "PFBF-30 " : 0,
            "FBF40 ": 0,
            "FBPI-20 ": 0 
            }

        folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"
        fileList = os.listdir(folder)

        for file in fileList:

            #need to get the total area of the interior of the slice of bread (without crust)
            img, scale = imgSearch(file.split('.')[0])
            canvasName = "crustCanvas/" + file.split('.')[0] + ".JPG"
            print(canvasName)
            canvas = cv2.imread(canvasName)

            """
            if canvas is None:
                canvasName = "crustCanvas/" + file[0:len(file)-7] + file[len(file)-6:len(file)-5] + ".JPG" 
                print(canvasName)
                canvas = cv2.imread(canvasName)

            if canvas is None:
                canvasName = "crustCanvas/" + file[0:len(file)-7] + " " + file[len(file)-7:len(file)-4] + ".JPG"
                print(canvasName)
                canvas = cv2.imread(canvasName)
            """
            breadInterior = removeCrust(img, canvas)       

            # thresholding with very high thresh to segment the image into the bread and the white background, finding contours on this segmentation, 
            # with the biggest being the external contour that contains the whole bread slice
            grey = cv2.cvtColor(breadInterior, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)[1]
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  
            contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

            
            breadArea = cv2.contourArea(contoursSorted[len(contoursSorted)-1])      #area of the biggest contour
            breadArea = breadArea/math.pow(scale,2)                                 #applying scale to shift from pixels to mm2

            data = pd.read_csv(folder+file)

            attributeData = list(data[attribute])     #this is the list of whatever attribute, ie luminence, for each airpocket
            areaData = list(data["area"])             #this is the list of areas (for each airpocket)

            for i in range(0, len(attributeData)-1):

                """
                if not attributeData[i] in dictX.get(file.split("(")[0]):    #checking if the luminence value is already in the list for that bread type (contained in dictX)
                    dictX.get(file.split("(")[0]).append(attributeData[i])      #adding luminence value to list if it's not already there
                    dictY[file.split("(")[0]].append(areaData[i]/breadArea)     #adding the corresponding percent area to the percent area set for that bread type in dictY
    
                else:
                    index = dictX.get(file.split("(")[0]).index(attributeData[i])       #finding position in dictX's values array where the attribute value already exists
                    dictY[file.split("(")[0]][index] += areaData[i]/breadArea           #adding the percent area to the value corresponding to that index position in the dictY area array
                """

                try:
                    cond = not attributeData[i] in dictX[file.split("(")[0]]
                except:
                    cond = not attributeData[i] in dictX[file[0:len(file)-5]]

                if cond:      #if value doesn't already exist
                    try:
                        dictX[file.split("(")[0]].append(attributeData[i])
                        dictY[file.split("(")[0]].append(areaData[i]/breadArea)
                    except:
                        dictX[file[0:len(file)-5]].append(attributeData[i])
                        dictY[file[0:len(file)-5]].append(areaData[i]/breadArea)

                else:
                    try:
                        index = dictX[file.split("(")[0]].index(attributeData[i])      #finding the index where the value already exists
                        dictY[file.split("(")[0]][index] += areaData[i]/breadArea          #adding the count to that index
                    except:
                        index = dictX[file[0:len(file)-5]].index(attributeData[i])
                        dictY[file[0:len(file)-5]][index] += areaData[i]/breadArea
 
            try:
                dictZ[file.split("(")[0]] += 1          #holds the number of images for each bread type
            except:
                dictZ[file[0:len(file)-5]] += 1       


        boxes = []
        labels = []
        #going to loop through the bread types and plot the boxplot of each. 
        for key in dictX.keys():

            #getting x and y data sets
            x = list(dictX[key])
            y = list(dictY[key])
            y[:] = [y/dictZ[key] for y in y]           #dividing each area value by the number of image files for that bread type

            box = []

            #boxplot method takes an array and plots the distribution of values within that array, so we need to get values from value, count format back to repeated values.
            #so I'm just going to add the value to a new array however many times the percent count, multiplied by 1 thousand because we need to round (ie 1%/0.01 becomes 10), for that value is. Should preserve the distribution that way
            #need for loop now to iterate over the x and y, and construct a new array which can be made into a boxplot
            for i in range(0,len(y)-1):        
                arr = np.full(math.floor(y[i]*1000), x[i], dtype="double")
                box.extend(arr)
            
            boxes.append(box)

            if key == "FBF-30":   
                labels.append("FBF-300s")
            elif key == "FBF-20":
                labels.append("FBF-200s")
            elif key == "FBF-10":
                labels.append("FBF-100s")
            else:      
                labels.append(key)              

        fig = plt.figure(figsize = (20,7))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(boxes)
        ax.set_xticklabels(labels)
        plt.title(attribute + " avg percent area")
        plt.savefig("dataAnalysis/boxplot/" + attribute + "Avg%Area.JPG", dpi=300, format='jpg')
        plt.close()





def size():
    """
    method to produce a csv (average, median, min, max, range, std) and boxplot of the air-pocket size of each bread type


    """
    dictX = {              
        "FBF5": [],
        "FBF25" : [],
        "FBF50" : [],
        "FBPI5" : [],
        "FBPI10" : [],
        "LF25" : [],
        "PF5" : [],
        "PF25" : [],
        "PF50" : [],
        "PPI5" : [],
        "WF" : [],
        "WF-100 " : [],
        "Control" : [],
        "FBF-30 " : [],
        "FBF-10" : [],
        "FBF-20" : [],
        "FBF-30" : [],
        "GFBF-30 " : [],
        "PGFBF-30 " : [],
        "PFBF-30 " : [],
        "FBF40 ": [],
        "FBPI-20 ": [] 
    }

    folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"
    fileList = os.listdir(folder)

    for file in fileList:
        data = pd.read_csv(folder+file)
        areaData = list(data["area"])           
        
        for airpocket in areaData:
            if not airpocket > 3:
                try:
                    dictX[file.split("(")[0]].append(airpocket)
                except:
                    dictX[file[0:len(file)-5]].append(airpocket)

    header = ["bread", "mean", "median", "min", "max", "std"]
    with open("outputData/airpocketSize.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  

    boxes = []
    labels = []
    index = 0

    for key in dictX.keys():

        if key == "FBF-30":   
            labels.append("FBF-300s")
        elif key == "FBF-20":
            labels.append("FBF-200s")
        elif key == "FBF-10":
            labels.append("FBF-100s")
        else:      
            labels.append(key)

        data = [labels[index], np.mean(dictX[key]), np.median(dictX[key]), np.min(dictX[key]), np.max(dictX[key]), np.std(dictX[key])]
        
        #writing to csv
        with open("outputData/airpocketSize.csv", 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
        
        box = []
        box.extend(dictX[key])
        boxes.append(box)

        index += 1




    fig = plt.figure(figsize = (20,15))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(boxes)
    ax.set_xticklabels(labels)
    plt.title("area of airpockets")
    plt.savefig("dataAnalysis/boxplot/area.JPG", dpi=300, format='jpg')
    plt.close()


def totalCount():
    """
    method to produce a csv (average, median, min, max, range, std) and boxplot of the air-pocket count of each bread type


    """
    dictX = {              
        "FBF5": [],
        "FBF25" : [],
        "FBF50" : [],
        "FBPI5" : [],
        "FBPI10" : [],
        "LF25" : [],
        "PF5" : [],
        "PF25" : [],
        "PF50" : [],
        "PPI5" : [],
        "WF" : [],
        "WF-100 " : [],
        "Control" : [],
        "FBF-30 " : [],
        "FBF-10" : [],
        "FBF-20" : [],
        "FBF-30" : [],
        "GFBF-30 " : [],
        "PGFBF-30 " : [],
        "PFBF-30 " : [],
        "FBF40 ": [],
        "FBPI-20 ": [] 
    }

    folder = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"
    fileList = os.listdir(folder)

    for file in fileList:
        data = pd.read_csv(folder+file)
        areaData = list(data["area"])           
        count = len(areaData)
    
        try:
            dictX[file.split("(")[0]].append(count)
        except:
            dictX[file[0:len(file)-5]].append(count)    


    header = ["bread", "mean", "median", "min", "max", "std"]
    with open("outputData/airpocketCount.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  

    boxes = []
    labels = []
    index = 0

    for key in dictX.keys():

        if key == "FBF-30":   
            labels.append("FBF-300s")
        elif key == "FBF-20":
            labels.append("FBF-200s")
        elif key == "FBF-10":
            labels.append("FBF-100s")
        else:      
            labels.append(key)

        data = [labels[index], np.mean(dictX[key]), np.median(dictX[key]), np.min(dictX[key]), np.max(dictX[key]), np.std(dictX[key])]

        #writing to csv
        with open("outputData/airpocketCount.csv", 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
        
        box = []
        box.extend(dictX[key])
        boxes.append(box)

        index += 1


    
    fig = plt.figure(figsize = (20,7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(boxes)
    ax.set_xticklabels(labels)
    plt.title("number of airpockets")
    plt.savefig("dataAnalysis/boxplot/totalCount.JPG", dpi=300, format='jpg')
    plt.close()


def condense():
    dir = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"

    dictX = {              
        "FBF5": [],
        "FBF25" : [],
        "FBF50" : [],
        "FBPI5" : [],
        "FBPI10" : [],
        "LF25" : [],
        "PF5" : [],
        "PF25" : [],
        "PF50" : [],
        "PPI5" : [],
        "WF" : [],
        "WF-100 " : [],
        "Control" : [],
        "FBF-30 " : [],
        "FBF-10" : [],
        "FBF-20" : [],
        "FBF-30" : [],
        "GFBF-30 " : [],
        "PGFBF-30 " : [],
        "PFBF-30 " : [],
        "FBF40 ": [],
        "FBPI-20 ": [] 
    }

    fileList = os.listdir(dir)

    for file in fileList:
        if file == ".DS_Store":
            continue

        #print(dir+file)
        try:
            df = pd.read_csv(dir+file)
        except:
            print(file)

        fileColumn = [file for x in range(len(df))]

        df.insert(loc=0, column="File", value=fileColumn)
        df.drop("count", axis = 1, inplace=True)

        try:
            dictX[file.split("(")[0]].append(df)
        except:
            dictX[file[0:len(file)-5]].append(df)

    for key in dictX.keys():
        combined = pd.concat(dictX[key], axis = 0)

        if key == "FBF-30":   
            label = "FBF-300s"
        elif key == "FBF-20":
            label = "FBF-200s"
        elif key == "FBF-10":
            label = "FBF-100s"
        else:      
            label = key

        combined.to_csv("outputData/condensedAirpockets/" + label + ".csv", index=False)




def areaVsCountHist():
    dir = "/Users/callum/Documents/breadBursaryProject/outputData/condensedAirpockets/"

    fileList = os.listdir(dir)

    cmap = matplotlib.cm.get_cmap('hsv')    #using hsv colour map

    index = 0


    fig = plt.figure(figsize = (10,15), dpi = 100)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    print(fileList)
    for file in fileList:
        if not re.search(".csv", file):
            continue
        print(file)
        df = pd.read_csv(dir+file)

        df = df.loc[df["area"] < 10]
        area = df["area"]


        
        counts,binEdges = np.histogram(area, bins = 50)

        standardisedCounts = [float(count)/float(len(df)) for count in counts]

        loggedCounts = [math.log(count) for count in standardisedCounts]

        binCenters = 0.5*(binEdges[1:] + binEdges[:-1])


        z = np.polyfit(binCenters, loggedCounts, 15)
        p = np.poly1d(z)

        percentile = index/25

        r,g,b,a = cmap(percentile)
        mappedColor = (r,g,b)

        ax1.plot(binCenters, loggedCounts, '-', label=file.split(".")[0], color=mappedColor)
        ax2.plot(binCenters, p(binCenters), '-', label=file.split(".")[0], color=mappedColor)



        index += 1
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")


    ax1.set_title("standard distribution of area")
    ax2.set_title("trend line fitted")

    ax1.set_xlabel("area")
    ax2.set_xlabel("area")
    ax1.set_ylabel("logged count")
    ax2.set_ylabel("logged count")


    plt.savefig("dataAnalysis/areaVsCount.JPG", format='jpg')


def findOutliers():
    dir = "/Users/callum/Documents/breadBursaryProject/outputData/airpocketData/"

    fileList = os.listdir(dir)

    for file in fileList:

        df = pd.read_csv(dir+file)

        areas = df["area"]

        for area in areas:
            if area > 300:
                print(file + ": " + str(area))


def main():

    avgCount()              
    avgPercentCount()      
    percentArea()          
    avgPercentArea()

    smallestToBiggest()    
    avgCountBoxplot()      
    avgPercentCountBoxplot()   
    percentAreaBoxplot()       
    avgPercentAreaBoxplot()    

    condense() 
    size() 
    totalCount()   
    areaVsCountHist()  

    #findOutliers()

main()


