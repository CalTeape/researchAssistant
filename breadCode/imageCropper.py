import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io, measure
import os



def crustCrop(img):
    """
    Method to crop an image of a slice of bread to remove the background, and be left with just the slice of bread on a blank white background.
    Works by using open cv in range method to find the contour of the bread. Then paints this contour in white onto a black mask, and uses bitiwse "and"
    operation to crop the image.

    :params img: 3d int array
        Array giving the b,g,r values for every pixel in the image
    
    :returns croppedImg: 3d int array
        The cropped image


    """
    mask = np.ones(img.shape, dtype=np.uint8)
    mask.fill(255)

    median = cv2.medianBlur(img, 3)                         #Apply median filter to reduce noise
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    hsv = cv2.inRange(hsv, (0, 45, 25), (27, 255, 255))
    thresh = cv2.threshold(hsv, 35, 255, 0)[1]          #Apply threshold (NOTE: thresh set low because just trying to 
                                                            #capture the whole slice at this stage)

    outerContours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]    #finding the external contours
    outerContours = sorted(outerContours, key=lambda x: cv2.contourArea(x))                  #sorting them from smallest to largest
    #cv2.drawContours(img, outerContours, len(outerContours) - 1, (0,255,0), 5)               #drawing the contour we're cropping along for visual aid/troubleshooting

    sliceContour = [outerContours[len(outerContours)-1]]            #this is an array of points which constitute the contour that borders the bread slice

    channelCount = img.shape[2]                                     #this is colour guf, just so it works with images that have different numbers of channels
    ignoreMaskColour = (0,)*channelCount                            #though we could have assumed RGB encoding so it doesn't matter

    cv2.fillPoly(mask, sliceContour, ignoreMaskColour)              #filling the contour in with black on the white mask

    croppedImg = cv2.bitwise_or(img, mask)                          #producing the cropped image with some sort of cv2 magic


    return croppedImg


