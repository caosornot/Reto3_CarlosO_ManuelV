#Import Libraries
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

# Create named window
cv.namedWindow('Line Follower')

# Create ON/OFF switch
def nothing(x):
    pass

cv.createTrackbar('Start - Stop', 'Line Follower', 0, 1, nothing)
cv.createTrackbar('Quit', 'Line Follower', 0, 1, nothing)
cv.setTrackbarPos('Start - Stop', 'Line Follower', 0)
cv.setTrackbarPos('Quit', 'Line Follower', 0)
# print(cv.getTrackBarPos('Quit','Line Follower'))


#Start video capture
videoPath = 'video_lab.mp4'

cap = cv.VideoCapture(videoPath)

#Define color boundaries for image processing
lowerBoundarie = np.array([35, 8, 0], dtype = 'uint8')
upperBoundarie = np.array([120, 255, 255], dtype = 'uint8')

# Display initial window
ret, frame = cap.read()
cv.imshow('Line Follower', frame)

# Loop through the video frames
while True:

    switch = cv.getTrackbarPos('Start - Stop', 'Line Follower')

    if switch:
        ret, frame = cap.read()
        img = frame
        imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV) 

        #Define Color Boundaries
        lowerBoundarie = np.array([35, 8, 0], dtype = 'uint8')
        upperBoundarie = np.array([120, 255, 255], dtype = 'uint8')
        #Compute mask for image
        mask = cv.inRange(imgHSV, lowerBoundarie, upperBoundarie)
        cv.imshow('Line Follower', frame)
        cv.imshow('Line Follower2', mask)

    # print(exitApp)
    k = cv.waitKey(5) & 0xFF
    if k == 27 or not ret:
        if not ret:
            print('exit due to not ret')
        elif exitApp:
            print('exit due to exitApp')
        break