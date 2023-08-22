from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import re
import random
import pandas as pd

from sklearn.model_selection import train_test_split # Import train_test_split function





def processFileString(file):
    result = ""
    for c in file:
        if c == " ":
            result += " "
        else:
            if not c.isalnum() and c != '.':
                result += "\\"
            result += c
    

    return result


def processFilePath(filepath):
    result = []
    if re.search("TwoFries", filepath):
            filePathSep =filepath.split("/")
            match = filePathSep[len(filePathSep)-1].split(".")[0]
            result.append(match[0:len(match)-3])
            result.append(match[0:len(match)-6] + match[len(match)-3:])
            #print(result)

    else:
        result.append(filepath)
        result.append(filepath)

    return result

def matchingCond(file, filepath):
    if re.search("No", filepath):
        return re.search("No", file)
    else:
        return not re.search("No", file)


def balanceDirectories():
    trainingGood = "fullData/testing/good/"
    trainingBad = "fullData/testing/bad/"
    
    dst_path = "fullData/testingOverflow/"

    good = os.listdir(trainingGood)
    bad = os.listdir(trainingBad)

    while len(bad) > len(good):
        os.rename(trainingBad + bad[0], dst_path + bad[0])
        bad.remove(bad[0]) 


def emptyData(dir):
    print(dir)
    if not re.search("DS_Store", dir):
        contents = os.listdir(dir)
        try:
            if re.search("jpg", contents[0]):
                for image in contents:
                    os.unlink(dir+image)
            else:
                for subdir in contents:
                    emptyData(dir + subdir + "/") 
        except:
            return 0


def distributeData(random_seed):

    emptyData("data/")

    col_names=["filename", "PEF", "Letter", "Potato", "Temperature", "Time"]
    feature_names = col_names[1:len(col_names)]

    feature_names = ["filename"]

    for i in range(20):
        col = "%k" + str(i)
        col_names.append(col)
        feature_names.append(col)


    target_variable = "Final majority vote"
    col_names.append(target_variable)

    dataset = pd.read_csv("csv/dataset.csv", header=0, names=col_names)

    print(len(dataset))

    x = dataset[feature_names]
    y = dataset[target_variable]

    trainingX, testingX, trainingY, testingY = train_test_split(x, y, random_state=random_seed, train_size = 0.6, stratify=y)
    testingX, validationX, testingY, validationY = train_test_split(testingX, testingY, random_state = random_seed, train_size=0.5, stratify = testingY)



    directories6 = []

    """
    dir = "processed/Healthy/segment/"
    healthies = os.listdir(dir)
    for image in healthies:
        directories6.append(dir+image)

    dir = "processed/Infected/segment/"
    infecties = os.listdir(dir)
    for image in infecties:
        directories6.append(dir+image)
    """

    dir = "fullData/"
    images = os.listdir(dir)
    for image in images:
        directories6.append(dir + image)

    print(len(directories6))

    #TRAINING
    targetFiles = trainingX["filename"]
    quality = trainingY
    qualityDict = dict(zip(targetFiles, quality))
    missing = list(targetFiles)

    zeroStringBadClass = []       #"<0", "00", "000"
    nonZeroBadClass = []            # >2 and 0
    goodClass = []                 # <= 2 and > 0


    for file in targetFiles:
        for filepath in directories6:
            #print(processFileString(file))
            if re.search(processFileString(file),filepath) and matchingCond(file, filepath):

            #    #MATCH
            #    goodOrBad = "bad/"
            #    
            #    if not re.search("000", qualityDict[file]) and not re.search("00", qualityDict[file]) and not re.search("<0", qualityDict[file]):
            #        if float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0:
            #            goodOrBad = "good/"

            #    img = cv2.imread(filepath)
            #    #print(file)
            #    cv2.imwrite("data/training/" + goodOrBad + file + ".jpg", img)  
            #    missing.remove(file)
            #    break


                #missing.remove(file)

                #directories6.remove(filepath)
                #print("FOUND MATCH BETWEEN: " + file + ", located at filepath: " + filepath)
                if re.search("000", qualityDict[file]) or re.search("00", qualityDict[file]) or re.search("<0", qualityDict[file]):
                    zeroStringBadClass.append(filepath)
                    continue
                
                elif float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0:
                    goodClass.append(filepath)
                    continue

                else:
                    nonZeroBadClass.append(filepath)
                    continue


    print("length of the zero string bad class training is: " + str(len(zeroStringBadClass)))
    print("length of the numeric bad class training is: " + str(len(nonZeroBadClass)))
    print("length of the numeric good class training is: " + str(len(goodClass)))



    for i in range(len(zeroStringBadClass)):        #NOTE THIS LOOP STRICTLY ASSUMES THAT THE NUMBER OF "00", "000", "<0" will be the smallest class out of the images
        img = cv2.imread(zeroStringBadClass[i])
        label = zeroStringBadClass[i].split("/")[1]
        cv2.imwrite("data/training/bad/" + label, img)

        img = cv2.imread(nonZeroBadClass[i])
        label = nonZeroBadClass[i].split("/")[1]
        cv2.imwrite("data/training/bad/" + label, img)


    j = 0
    while j <= 2*len(zeroStringBadClass):
        img = cv2.imread(goodClass[j])
        label = goodClass[j].split("/")[1]
        cv2.imwrite("data/training/good/" + label, img)
        j += 1




    #"""
    #VALIDATION
    targetFiles = validationX["filename"]
    quality = validationY
    qualityDict = dict(zip(targetFiles, quality))
    missing.extend(targetFiles)


    zeroStringBadClass = []        #"<0", "00", "000"
    nonZeroBadClass = []            # >2 and 0
    goodClass = []                  # <= 2 and > 0


    for file in targetFiles:
        for filepath in directories6:
            if re.search(processFileString(file),filepath) and matchingCond(file, filepath):
            #    goodOrBad = "bad/"
            #    if not re.search("000", qualityDict[file]) and not re.search("00", qualityDict[file]) and not re.search("<0", qualityDict[file]):
            #        if float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0:
            #            goodOrBad = "good/"

            #    img = cv2.imread(filepath)
            #    #print(file)
            #    cv2.imwrite("data/validation/" + goodOrBad + file + ".jpg", img)
            #    missing.remove(file)
            #    break

                if re.search("000", qualityDict[file]) or re.search("00", qualityDict[file]) or re.search("<0", qualityDict[file]):
                    zeroStringBadClass.append(filepath)
                    continue
                
                elif float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0:
                    goodClass.append(filepath)
                    continue

                else:
                    nonZeroBadClass.append(filepath)
                    continue

    print("length of the zero string bad class validation is : " + str(len(zeroStringBadClass)))



    for i in range(len(zeroStringBadClass)):        #NOTE THIS LOOP STRICTLY ASSUMES THAT THE NUMBER OF "00", "000", "<0" will be the smallest class out of the images
        img = cv2.imread(zeroStringBadClass[i])
        label = zeroStringBadClass[i].split("/")[1]
        cv2.imwrite("data/validation/bad/" + label, img)

        img = cv2.imread(nonZeroBadClass[i])
        label = nonZeroBadClass[i].split("/")[1]
        cv2.imwrite("data/validation/bad/" + label, img)


    j = 0
    while j <= 2*len(zeroStringBadClass):
        img = cv2.imread(goodClass[j])
        label = goodClass[j].split("/")[1]
        cv2.imwrite("data/validation/good/" + label, img)
        j += 1


   
    #TESTING
    targetFiles = testingX["filename"]
    quality = testingY
    qualityDict = dict(zip(targetFiles, quality))
    missing.extend(targetFiles)


    zeroStringBadClass = []         #"<0", "00", "000"
    nonZeroBadClass = []            # >2 and 0
    goodClass = []                  # <= 2 and > 0

    for file in targetFiles:
        for filepath in directories6:
            if re.search(processFileString(file), filepath) and matchingCond(file, filepath):
            #    goodOrBad = "bad/"
            #    if not re.search("000", qualityDict[file]) and not re.search("00", qualityDict[file]) and not re.search("<0", qualityDict[file]):
            #        if float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0:
            #            goodOrBad = "good/"

            #    img = cv2.imread(filepath)
            #    cv2.imwrite("data/testing/" + goodOrBad + file + ".jpg", img)
            #    missing.remove(file)
            #    break
    



                if re.search("000", qualityDict[file]) or re.search("00", qualityDict[file]) or re.search("<0", qualityDict[file]):
                    zeroStringBadClass.append(filepath)
                    continue
                
                elif float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0:
                    goodClass.append(filepath)
                    continue

                else:
                    nonZeroBadClass.append(filepath)
                    continue

    print("length of the zero string bad class testing is : " + str(len(zeroStringBadClass)))



    for i in range(len(zeroStringBadClass)):        #NOTE THIS LOOP STRICTLY ASSUMES THAT THE NUMBER OF "00", "000", "<0" will be the smallest class out of the images
        img = cv2.imread(zeroStringBadClass[i])
        label = zeroStringBadClass[i].split("/")[1]
        cv2.imwrite("data/testing/bad/" + label, img)

        img = cv2.imread(nonZeroBadClass[i])
        label = nonZeroBadClass[i].split("/")[1]
        cv2.imwrite("data/testing/bad/" + label , img)


    j = 0
    while j <= 2*len(zeroStringBadClass):
        img = cv2.imread(goodClass[j])
        label = goodClass[j].split("/")[1]
        cv2.imwrite("data/testing/good/" + label, img)
        j += 1



