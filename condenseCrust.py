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


def condense():
    fp = "/Users/callum/Documents/breadBursaryProject/outputData/crustData.csv"

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
        "FBF40 " : [],
        "FBF-30 " : [],
        "FBF-10" : [],
        "FBF-20" : [],
        "FBF-30" : [],
        "GFBF-30 " : [],
        "PGFBF-30 " : [],
        "PFBF-30 " : [],
        "FBPI-20 " : []
    }


    df = pd.read_csv(fp)

    for index, row in df.iterrows():
        #print(row)
        try:
            dictX[row[0].split("(")[0]].append(row)
        except:
            filename = row[0][:len(row[0])-5]
            dictX[filename].append(row)
            #print(filename)

    for key in dictX.keys():
        combined = pd.concat(dictX[key], axis = 1).transpose()

        if key == "FBF-30":   
            label = "FBF-300s"
        elif key == "FBF-20":
            label = "FBF-200s"
        elif key == "FBF-10":
            label = "FBF-100s"
        else:      
            label = key

        combined.to_csv("outputData/condensedCrust/" + label + ".csv", index=False)

def main():
    condense()

main()