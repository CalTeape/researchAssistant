from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
#import cv2
import imageio.v2 as iio
import os
import re
import random
import pandas as pd

from rotateAndFlip import iterate
from rotateAndFlip import rotate
from rotateAndFlip import center

from sklearn.model_selection import train_test_split # Import train_test_split function


"""
NEED TO CHANGE THE WAY THIS WORKS:

HAVE SOME TARGET NUMBER OF IMAGES, and only apply as many rotations as neccessary to reach target. So big classes may 
only have  a few, but small classes need to be fleshed out with many to have the same size as the big classes.

"""
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


def balanceDirectories(targetDir):

    zero = os.listdir(targetDir + "zeroString")
    good = os.listdir(targetDir + "good")
    bad = os.listdir(targetDir + "bad")

    dirStrings = ["zeroString/", "good/", "bad/"]
    dirs = [zero, good, bad]


    target = max(len(zero), len(good), len(bad))*10

    for k in range(3):
        count = len(dirs[k])
        index = 0
        numRotations = math.ceil(target/len(dirs[k]))

        while count < target:
            if index > len(dirs[k]):
                index = 0
            file = dirs[k][index]
            print(targetDir + dirStrings[k]+file)
            img = iio.imread(targetDir + dirStrings[k] + file)
            img = center(img)

            for i in range(1,numRotations):
                theta = int(360*(i/numRotations))
                rotated = rotate(img, theta)
                iio.imwrite(targetDir + dirStrings[k] + file.split(".")[0]+ "(" + str(theta) + ").jpg", rotated)
                count += 1

            index += 1







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

    dataset = pd.read_csv("dataset.csv", header=0, names=col_names)

    print(len(dataset))

    x = dataset[feature_names]
    y = dataset[target_variable]

    trainingX, testingX, trainingY, testingY = train_test_split(x, y, random_state=random_seed, train_size = 0.6, stratify=y)
    testingX, validationX, testingY, validationY = train_test_split(testingX, testingY, random_state = random_seed, train_size=0.5, stratify = testingY)



    directories6 = []

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

    zeroString = []       #"<0", "00", "000"
    badClass = []            # >2 and 0
    goodClass = []                 # <= 2 and > 0


    for file in targetFiles:
        for filepath in directories6:
            #print(processFileString(file))
            if re.search(processFileString(file),filepath) and matchingCond(file, filepath):

                if re.search("000", qualityDict[file]) or re.search("00", qualityDict[file]):
                    zeroString.append(filepath)
                    continue

                elif re.search("<0", qualityDict[file]):
                    zeroString.append(filepath)
                    continue

                elif float(qualityDict[file]) <= 0.5:
                    zeroString.append(filepath)
                    continue

                elif float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0.5:
                    goodClass.append(filepath)
                    continue

                else:
                    badClass.append(filepath)
                    continue


    print("length of zero class training is: " + str(len(zeroString)))
    print("length of the good class training is: " + str(len(goodClass)))
    print("length of the bad class training is: " + str(len(badClass)))

    for i in range(max(len(zeroString), len(goodClass), len(badClass))):
        if i < len(zeroString):
            img = iio.imread(zeroString[i])
            label = zeroString[i].split("/")[1]
            iio.imwrite("data/training/zeroString/" + label, img)

        if i < len(badClass):
            img = iio.imread(badClass[i])
            label = badClass[i].split("/")[1]
            iio.imwrite("data/training/bad/" + label, img)

        if i < len(goodClass):
            img = iio.imread(goodClass[i])
            label = goodClass[i].split("/")[1]
            iio.imwrite("data/training/good/" + label, img)


    #"""
    #VALIDATION
    targetFiles = validationX["filename"]
    quality = validationY
    qualityDict = dict(zip(targetFiles, quality))
    missing.extend(targetFiles)


    zeroString = []       #"<0", "00", "000"
    subZero = []
    zero = []
    badClass = []            # >2 and 0
    goodClass = []                 # <= 2 and > 0


    for file in targetFiles:
        for filepath in directories6:
            #print(processFileString(file))
            if re.search(processFileString(file),filepath) and matchingCond(file, filepath):

                if re.search("000", qualityDict[file]) or re.search("00", qualityDict[file]):
                    zeroString.append(filepath)
                    continue

                elif re.search("<0", qualityDict[file]):
                    zeroString.append(filepath)
                    continue

                elif float(qualityDict[file]) <= 0.5:
                    zeroString.append(filepath)
                    continue

                elif float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0.5:
                    goodClass.append(filepath)
                    continue

                else:
                    badClass.append(filepath)
                    continue

    print("length of zero class val is: " + str(len(zeroString)))
    print("length of the good class val is: " + str(len(goodClass)))
    print("length of the bad class val is: " + str(len(badClass)))

    for i in range(max(len(zeroString), len(goodClass), len(badClass))):
        if i < len(zeroString):
            img = iio.imread(zeroString[i])
            label = zeroString[i].split("/")[1]
            iio.imwrite("data/validation/zeroString/" + label, img)

        if i < len(badClass):
            img = iio.imread(badClass[i])
            label = badClass[i].split("/")[1]
            iio.imwrite("data/validation/bad/" + label, img)

        if i < len(goodClass):
            img = iio.imread(goodClass[i])
            label = goodClass[i].split("/")[1]
            iio.imwrite("data/validation/good/" + label, img)

    #TESTING
    targetFiles = testingX["filename"]
    quality = testingY
    qualityDict = dict(zip(targetFiles, quality))
    missing.extend(targetFiles)

    zeroString = []       #"00", "000"
    subZero = []
    zero = []
    badClass = []            # >2 and 0
    goodClass = []                 # <= 2 and > 0


    for file in targetFiles:
        for filepath in directories6:
            #print(processFileString(file))
            if re.search(processFileString(file),filepath) and matchingCond(file, filepath):

                if re.search("000", qualityDict[file]) or re.search("00", qualityDict[file]):
                    zeroString.append(filepath)
                    continue

                elif re.search("<0", qualityDict[file]):
                    zeroString.append(filepath)
                    continue

                elif float(qualityDict[file]) <= 0.5:
                    zeroString.append(filepath)
                    continue

                elif float(qualityDict[file]) <= 2 and float(qualityDict[file]) > 0.5:
                    goodClass.append(filepath)
                    continue

                else:
                    badClass.append(filepath)
                    continue

    print("length of zero class testing is: " + str(len(zeroString)))
    print("length of the good class testing is: " + str(len(goodClass)))
    print("length of the bad class testing is: " + str(len(badClass)))

    for i in range(max(len(zeroString), len(goodClass), len(badClass))):
        if i < len(zeroString):
            img = iio.imread(zeroString[i])
            label = zeroString[i].split("/")[1]
            iio.imwrite("data/testing/zeroString/" + label, img)

        if i < len(badClass):
            img = iio.imread(badClass[i])
            label = badClass[i].split("/")[1]
            iio.imwrite("data/testing/bad/" + label, img)

        if i < len(goodClass):
            img = iio.imread(goodClass[i])
            label = goodClass[i].split("/")[1]
            iio.imwrite("data/testing/good/" + label, img)


    balanceDirectories("data/training/")
    balanceDirectories("data/testing/")
    balanceDirectories("data/validation/")

def fullDataGen():

    directories6 = []

    dir = "processed/Healthy/segment/"
    healthies = os.listdir(dir)
    for image in healthies:
        directories6.append(dir+image)

    dir = "processed/Infected/segment/"
    infecties = os.listdir(dir)
    for image in infecties:
        directories6.append(dir+image)
    
    dataset = pd.read_csv("dataset.csv", header=0)

    print(len(dataset))

    filenames = dataset["filename"]

    for file in filenames:
        for filepath in directories6:
            if re.search(processFileString(file), filepath) and matchingCond(file, filepath):
                img = iio.imread(filepath)
                iio.imwrite("fullData/" + file + ".jpg", img)
                break





searchFiles=[]
def recurse(target):

    contents = os.listdir(target)

    for dir in contents:
        if re.search(".csv", dir) or re.search("graysegment", target+dir):
            continue

        elif re.search(".JPG", dir) and re.search("segment", target+dir):
            searchFiles.extend([target + x for x in contents])
            break

        elif re.search(".JPG", dir):
            continue

        else:
            recurse(target+dir+"/")


def getAllImages():

    emptyData("allImages/")

    filepaths = []
    searchDirs = ["allTargets/TwoFries/", "allTargets/Healthy/", "allTargets/Infected/", "processed/"]

    df = pd.read_excel("master_sheet.xlsx")

    targets = list(df["filename"])

    targetsCopy = list(df["filename"])

    count = 0

    while len(targets)>0 and count < 4:

        recurse(searchDirs[count])

        print("WE ARE MISSING: " + str(len(targets)))

        for file in targets:
            for filepath in searchFiles:
                if re.search(processFileString(file), filepath) and matchingCond(file, filepath):
                    #print("FOUND MATCH: " + file + "    WITH: " + filepath)
                    filepaths.append(filepath)
                    if file in targetsCopy:
                        targetsCopy.remove(file)



        found = len(targets) - len(targetsCopy)
        print("WE FOUND: " + str(found))
        targets = np.copy(targetsCopy)
        count += 1

    print("Missing: " + str(targets))

    print(len(filepaths))


    for filepath in filepaths:
        comps = filepath.split("/")
        label = comps[len(comps)-1]
        img = iio.imread(filepath)
        iio.imwrite("allImages/"+label, img)


getAllImages()




