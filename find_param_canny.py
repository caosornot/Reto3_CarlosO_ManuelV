import cv2 as cv
import numpy as np

#Load image & create window
img = cv.imread('5.jpeg')
cv.namedWindow('Parameters')
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

def nothing(x):
    pass

#On/Off switch for HSV
cv.createTrackbar('switch', 'Parameters',0,1,nothing)

#Create trackbars for canny algorithm
cv.createTrackbar('Canny_lower','Parameters', 0, 1000, nothing)
cv.createTrackbar('Canny_upper','Parameters', 0, 1000, nothing)

while(1):
    
    # Get trackbar positions for the five trackbars
    s1 = cv.getTrackbarPos('switch','Parameters')

    lowerCanny = cv.getTrackbarPos('Canny_lower', 'Parameters')
    upperCanny = cv.getTrackbarPos('Canny_lower', 'Parameters')
    
    if s1:
        #Define Color Boundaries
        lowerBoundarie = np.array([35, 8, 0], dtype = 'uint8')
        upperBoundarie = np.array([120, 255, 255], dtype = 'uint8')
        #Compute mask for image
        mask = cv.inRange(imgHSV, lowerBoundarie, upperBoundarie)
        out = cv.bitwise_and(imgHSV, imgHSV, mask=mask)
        edges = cv.Canny(mask, 200, 500)
        cv.imshow('Parameters2', edges)
        cv.imshow('Original', img)
    else:
        cv.imshow('Parameters', imgHSV)
        
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    