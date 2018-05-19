import cv2 as cv
import numpy as np

#Load image & create window
img = cv.imread('5.jpeg')
cv.namedWindow('Parameters')


def nothing(x):
    pass

#On/Off switch for HSV
cv.createTrackbar('switch', 'Parameters',0,1,nothing)

#Create trackbars for lower limits
cv.createTrackbar('B_lower', 'Parameters', 0,255, nothing)
cv.createTrackbar('G_lower', 'Parameters', 0, 255, nothing)
cv.createTrackbar('R_lower', 'Parameters', 0, 255, nothing)

#Create trackbars or upper limits
cv.createTrackbar('B_upper', 'Parameters', 0,255, nothing)
cv.createTrackbar('G_upper', 'Parameters', 0, 255, nothing)
cv.createTrackbar('R_upper', 'Parameters', 0, 255, nothing)

while(1):
    
    # Get trackbar positions for the five trackbars
    s1 = cv.getTrackbarPos('switch','Parameters')
   
    lowerBlue = cv.getTrackbarPos('B_lower', 'Parameters')
    lowerGreen = cv.getTrackbarPos('G_lower', 'Parameters')
    lowerRed = cv.getTrackbarPos('R_lower', 'Parameters')

    upperBlue = cv.getTrackbarPos('B_upper', 'Parameters')
    upperGreen = cv.getTrackbarPos('G_upper', 'Parameters')
    upperRed = cv.getTrackbarPos('R_upper', 'Parameters')
    
    if s1:
        lower = np.array([lowerBlue, lowerGreen, lowerRed], dtype = 'uint8')
        upper = np.array([upperBlue, upperGreen, upperRed], dtype = 'uint8')
        mask = cv.inRange(img, lower, upper)
        out = cv.bitwise_and(img, img, mask=mask)
        cv.imshow('Parameters2', out)
        cv.imshow('Original', img)
    
    else:
        cv.imshow('Parameters', img)
        
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    