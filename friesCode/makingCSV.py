import pandas as pd
import numpy as np
import csv
import os
import operator as op
import re


def combine():
    dir = "/Users/callum/Documents/frenchFriesBursaryProject/[NEW] French fries csv files to rate/Combine/"

    fileList = os.listdir(dir)
    arr = []
    for file in fileList:
        df = pd.read_csv(dir+file)
        arr.append(df)

    combined = pd.concat(arr, axis = 0)

    combined.to_csv("combined.csv", index=False)



def compressCombine():
    df = pd.read_csv("combined.csv")

    columnPEF = list(df["PEF"])

    PEF = []
    letter = []

    for r in columnPEF:
       if re.search("No", r):
          PEF.append("No")
       else:
          PEF.append("Yes")
    
       letter.append(r[len(r)-1])

    df.drop('PEF', axis=1, inplace=True)

    df.insert(loc=1, column="PEF", value=PEF)
    df.insert(loc=2, column="Letter", value=letter)

    df.to_csv("combined.csv", index=False)



def combineCheck():
    df = pd.read_csv("combined.csv")
    filenames = list(df["filename"])
    fileSet = set(filenames)

    if not len(fileSet) == len(filenames):
        print("combined contains duplicates")
    else:
        print("combined does not contain duplicates")

    letters = list(df["Letter"])
    for letter in letters:
        if not letter == 'A' and not letter =='B' and not letter == 'C' and not letter == 'D':
            print(letter)




def reference():
    file = "/Users/callum/Documents/frenchFriesBursaryProject/[NEW] French fries csv files to rate/Reference_French fries score result on selected images.xlsx"

    dict_df = pd.read_excel(file, sheet_name=['Healthy', 'Infected'])

    healthy_df = dict_df.get('Healthy')
    infected_df = dict_df.get('Infected')

    arr = [healthy_df, infected_df]
    df = pd.concat(arr, axis = 0)



    filename = list(df["filename"])

    #for some reason, alot of the filenames in the graded csv contain a space at the end, so have to remove otherwise inner join doesn't work
    for i in range(len(filename)):
        if filename[i][len(filename[i])-1] == ' ':
            filename[i] = filename[i][0:len(filename[i])-1]


    vote = list(df["Final majority vote"])

    reference = pd.DataFrame(zip(filename, vote), columns = ["filename", "Final majority vote"])

    reference.to_csv("reference.csv", index=False)





def innerJoin():
    dfCombined = pd.read_csv("combined.csv")
    dfReference = pd.read_csv("reference.csv")

    joined = dfCombined.merge(dfReference, on='filename')

    joined.to_csv("dataset.csv", index=False)




def main():
    #combine()
    #compressCombine()
    #combineCheck()
    reference()
    innerJoin()

main()
    
