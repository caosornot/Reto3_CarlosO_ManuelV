# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:35:06 2018

@author: DELL
"""

import cv2 as cv
import numpy as np

#Load image & create window
img = cv.imread('1.jpg')
cv.namedWindow('Color Thresholds')
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

def nothing(x):
    pass

#On/Off switch
cv.createTrackbar('switch', 'Color Thresholds',0,1,nothing)

#Create trackbars for lower limits
cv.createTrackbar('H_lower', 'Color Thresholds', 0,255, nothing)
cv.createTrackbar('S_lower', 'Color Thresholds', 0, 255, nothing)
cv.createTrackbar('V_lower', 'Color Thresholds', 0, 255, nothing)

#Create trackbars or upper limits
cv.createTrackbar('H_upper', 'Color Thresholds', 0,255, nothing)
cv.createTrackbar('S_upper', 'Color Thresholds', 0, 255, nothing)
cv.createTrackbar('V_upper', 'Color Thresholds', 0, 255, nothing)



while(1):
    
    # Get trackbar positions for the five trackbars
    s1 = cv.getTrackbarPos('switch','Color Thresholds')
   
    lowerBlue = cv.getTrackbarPos('H_lower', 'Color Thresholds')
    lowerGreen = cv.getTrackbarPos('S_lower', 'Color Thresholds')
    lowerRed = cv.getTrackbarPos('V_lower', 'Color Thresholds')

    upperBlue = cv.getTrackbarPos('H_upper', 'Color Thresholds')
    upperGreen = cv.getTrackbarPos('S_upper', 'Color Thresholds')
    upperRed = cv.getTrackbarPos('V_upper', 'Color Thresholds')
    
    if s1:
        lower = np.array([lowerBlue, lowerGreen, lowerRed], dtype = 'uint8')
        upper = np.array([upperBlue, upperGreen, upperRed], dtype = 'uint8')
        mask = cv.inRange(imgHSV, lower, upper)
        out = cv.bitwise_and(imgHSV, imgHSV, mask=mask)
        cv.imshow('Color Thresholds2', out)
        cv.imshow('Original', img)
    else:
        cv.imshow('Color Thresholds', imgHSV)
        
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    