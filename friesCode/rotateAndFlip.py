import cv2 
import numpy as np
import os
import re
import math
import matplotlib.pyplot as plt

def center(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY_INV)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))

    x,y,w,h = cv2.boundingRect(contours[len(contours)-1])

    deltaX = (6000/2) - (x+w/2)
    deltaY = (6000/2) - (y+h/2)

    M = np.array([[1,0,deltaX],[0,1,deltaY]])

    centered = cv2.warpAffine(img, M, (6000,6000), borderValue=(255,255,255))

    return centered
    

def rotate(image, angle):
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, (6000,6000), borderValue=(255,255,255), flags=cv2.INTER_LINEAR)

    return result



def recurse(dir, outputDir):
    print(dir)
    if not re.search("DS_Store", dir):
        contents = os.listdir(dir)
        if re.search("jpg", contents[0]):
            for image in contents:
                print(dir + image)
                img = cv2.imread(dir+image)
                img = center(img)

                theta = 0
                while theta < 360:
                    rotated = rotate(img, theta)
                    if theta == 0:
                        cv2.imwrite(outputDir+image, rotated)
                        flip_v = cv2.flip(rotated, 0)
                        flip_h = cv2.flip(rotated, 1)
                        cv2.imwrite(outputDir+image.split(".")[0]+"(flipped_h).jpg", flip_h)
                        cv2.imwrite(outputDir+image.split(".")[0]+"(flipped_v).jpg", flip_v)
                    else:
                        cv2.imwrite(outputDir+image.split(".")[0]+"(" + str(theta) + ").jpg", rotated)
                    theta += 30


        else:
            for subdir in contents:
                recurse(dir + subdir + "/", outputDir + subdir + "/")

def main():
    recurse("data/training/bad/", "fullData/training/bad/")

main()