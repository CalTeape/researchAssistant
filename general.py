from typing import List
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from skimage import io, measure
import random
import os
import re
import numpy as np
import csv
import numpy as np
from pathlib import Path
import math
import json
import os
from operator import itemgetter


def plot_images(file, images: List[np.ndarray], figure_title:str = "", subplot_titles:List[str]=[], cmap: str = "viridis"):
    """
    Plot a series of images in a grid using matplotlib and subplots
    Useful to show interpretations of different filters in conv-nets

    :param images: List[np.ndarray]
            The list of images to be plotted
            Images (items of the array) should be square, etc.. and ar passed directly to imshow
            The list itself can be any length (even non-square)

    :param figure_title: str
            The title of the entire figure

    :param subplot_titles: List[str]
            Title each image. Defaults to empty array.
            If empty, no titles are added

    :param cmap: str
            The colour map to use for the images


    """


    total_cols = int(len(images)**0.5)
    total_rows = len(images) // total_cols
    if len(images) % total_cols != 0: total_rows += 1

    fig = plt.figure(figsize=(10,10))
    fig.suptitle(figure_title)
    for i in range(0,len(images)):
        ax = fig.add_subplot(total_rows, total_cols, i+1)
        ax.imshow(images[i], cmap=cmap)
        ax.axis("off")
        if i < len(subplot_titles):
            ax.set_title(subplot_titles[i], fontsize=8)

    plt.tight_layout()
    plt.show()
    #plt.savefig("outputImages/airpockets/" + file.split('.')[0] + ".JPG", format='jpg')


def adjustContrast(img, vibrance):
    """
    Method to adjust the contrast of an image.
    From Stack Overflow.

    :params img: 3d int array
        Array giving the b,g,r values for every pixel in the image
    :params vibrance: decimal
        The factor we want to increase the contrast by (ie 1.4 for 140% more contrast)

    :returns result: 3d int array
        The contrast-adjusted image


    """
    # convert image to hsv colorspace as floats
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)


    # create 256 element non-linear LUT for sigmoidal function
    # see https://en.wikipedia.org/wiki/Sigmoid_function
    xval = np.arange(0, 256)
    lut = (255*np.tanh(vibrance*xval/255)/np.tanh(1)+0.5).astype(np.uint8)

    # apply lut to saturation channel
    new_s = cv2.LUT(s,lut)

    # combine new_s with original h and v channels
    new_hsv = cv2.merge([h,new_s,v])

    # convert back to BGR
    result =  cv2.cvtColor(new_hsv,  cv2.COLOR_HSV2BGR)
    return result


def rgb_to_hsv(r, g, b):
    """
    Method to convert a tupule from r,g,b colour space to h,s,v.
    From https://www.geeksforgeeks.org

    :params r: int
        red channel
    :params g: int
        green channel
    :params b: int
        blue channel

    :returns h,s,v: int
        hue, saturation, value
    """
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.
 
    if cmax == cmin:     # if cmax and cmax are equal then h = 0
        h = 0
    elif cmax == r:         # if cmax equal r then compute h
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:    # if cmax equal g then compute h
        h = (60 * ((b - r) / diff) + 120) % 360
    elif cmax == b:    # if cmax equal b then compute h
        h = (60 * ((r - g) / diff) + 240) % 360
    if cmax == 0:    # if cmax equal zero
        s = 0
    else:
        s = (diff / cmax) * 100

    v = cmax * 100    # compute v
    return h, s, v



def rgb2linDot(value):
    '''
    From Veronica
    '''
    if value <= 0.04045:
        return value / 12.92
    else:
        return pow(((value + 0.055)/1.055), 2.4)

def lumin(b,g,r):
    '''
    From Veronica
    '''
    r = r / 255
    g = g / 255
    b = b / 255
    Y = (0.2126 * rgb2linDot(r) + 0.7152 * rgb2linDot(g) + 0.0722 * rgb2linDot(b))
    if Y <= (216/24389):
        return Y * (24389 / 27)
    else:
        return pow(Y, (1 / 3)) * 116 - 16



def resize(image, scale=0.4):
    """
    method to make the image roughly 0.4^2 big as it was.

    :params image: 3d int array
        Array giving the b,g,r values for every pixel in the image
    :params scale: decimal
        The factor to reduce the image dimensions by
    
    :returns image_scaled: 3d int array
        The scaled image


    """
    dimension = image.shape
    height = dimension[0]
    width = dimension[1]
    dimension_scaled = (int(scale*width), int(scale*height))

    image_scaled = cv2.resize(image, dimension_scaled)
    return image_scaled




def getScale(file, img):
    """
    Method to get the scale of the image in pixels per mm.
    Works by using open cv in range to detect the green in the image, which is always the ruler. Then gets the bounding box on this contour 
    and uses pythagoras to figure out the length of the ruler in pixels. Divides this by the 152mm of the ruler to give pixels per mm.

    :params file: String
        Name of the image file
    :params img: 3d int array
        Array giving the b,g,r values for every pixel in the image

    :returns scale: int
        The number of pixels per mm.


    """
    if re.search("PF50", file): #for some reason pf50s are a special case
        return 4

    img = resize(img)

    #grab the green ruler
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #mask the green
    #mask = cv2.inRange(hsv, (40, 50, 0), (80, 255,255))

    mask = cv2.inRange(hsv, (40, 50, 0), (100, 255,255))

    ## slice the green
    ker = np.ones((3,3), dtype="uint8")
    mask = cv2.dilate(mask, ker, 1)             #dilation required because the contour captured is often a little patchy, dilation joins all bits of the ruler together


    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x)) #sorting the contours from smallest to largest (largest contour should be the ruler)

    rect = cv2.minAreaRect(contoursSorted[len(contoursSorted)-1])   #getting bounding box, using "minAreaRect" to allow rotation
    box = cv2.boxPoints(rect)                                       #turning into array of four corners
    box = np.int0(box)                                              #casting to numpy array

    #cv2.drawContours(img, [box], 0, (0,0,255), 5)                   #plotting for demonstration

    #using pythagoras to get the height and width of the ruler


    lengthIfVertical = math.sqrt(math.pow(box[1][0] - box[0][0],2) + math.pow(box[1][1]-box[0][1],2))
    lengthIfHorizontal = math.sqrt(math.pow(box[2][0] - box[1][0],2) + math.pow(box[2][1] - box[1][1], 2))
    length = max(lengthIfVertical, lengthIfHorizontal)      #taking the max of the two to account for cases when the orientation of the ruler is different
    scale = length/152     #pixels per mm

    if abs(scale-3.9) > 1:
        print(file)
        print(scale)

    return scale

    