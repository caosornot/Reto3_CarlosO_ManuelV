#Import Libraries
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import imutils
import math
import serial, time


# Create named window
cv.namedWindow('Line Follower Original')

# Create ON/OFF switch
def nothing(x):
    pass

cv.createTrackbar('Start - Stop', 'Line Follower Original', 0, 1, nothing)
cv.createTrackbar('Quit', 'Line Follower Original', 0, 1, nothing)
cv.setTrackbarPos('Start - Stop', 'Line Follower Original', 0)
cv.setTrackbarPos('Quit', 'Line Follower Original', 0)
# print(cv.getTrackBarPos('Quit','Line Follower'))


#Start video capture
videoPath = 'hallway-sim-morning.mp4'

cap = cv.VideoCapture(videoPath)

#Define color boundaries for image processing
lowerBoundarie = np.array([35, 8, 0], dtype = 'uint8')
upperBoundarie = np.array([120, 255, 255], dtype = 'uint8')

# Display initial window
ret, frame = cap.read()
cv.imshow('Line Follower Original', frame)

#Define width of image

# Loop through the video frames
while True:

    switch = cv.getTrackbarPos('Start - Stop', 'Line Follower Original')

    if switch:
        ret, frame = cap.read()
        img = frame
        imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
        angle = 0
        imgHSV = imutils.rotate(imgHSV, angle)
        imgHSV = imutils.resize(imgHSV, width=320, height=240)
        w = imgHSV.shape [1]
        h = imgHSV.shape [0]

        #Define Color Boundaries
        lowerBoundarie = np.array([50, 35, 112], dtype = 'uint8')
        upperBoundarie = np.array([105, 129, 131], dtype = 'uint8')

        #Apply mask and compute x & y coordinates (into a single array) of white pixels after applying the mask
        mask = cv.inRange(imgHSV, np.array(lowerBoundarie), np.array(upperBoundarie))
        pixelInfo = cv.inRange(imgHSV, np.array(lowerBoundarie), np.array(upperBoundarie))
        array = np.nonzero(pixelInfo)
        transposedArray = np.transpose(array)

        #Separate x & y coordinates (different arrays)
        X = transposedArray[: , 1]
        y = transposedArray[: , 0]

        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)     

        #Start RANSAC algorithm
        # Robustly fit first linear model with RANSAC algorithm
        firstRansac = linear_model.RANSACRegressor(residual_threshold=10)
        firstRansac.fit(X, y)
        firstinlier_mask = firstRansac.inlier_mask_
        firstoutlier_mask = np.logical_not(firstinlier_mask)

        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        first_line_y_ransac = firstRansac.predict(line_X)

        #Calculate and print slope and intercept for the first line.
        firstSlope = firstRansac.estimator_.coef_
        firstIntercept = firstRansac.estimator_.intercept_
        # print (firstSlope)
        # print(firstIntercept)

        #Delete first line
        i = 0
        deletePointsX = []
        deletePointsY = []

        while i<len(X): 

            if abs((firstSlope*X[i])-(y[i])+(firstIntercept))/(math.sqrt((firstSlope)**2)+1) < 50:
                deletePointsX.append(i)
                deletePointsY.append(i)
            i += 1

        X = np.delete(X, deletePointsX)
        y = np.delete(y, deletePointsY)

        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # Robustly fit Second linear model with RANSAC algorithm
        secondRansac = linear_model.RANSACRegressor(residual_threshold=10)
        secondRansac.fit(X, y)
        secondinlier_mask = secondRansac.inlier_mask_
        secondoutlier_mask = np.logical_not(secondinlier_mask)

        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        second_line_y_ransac = secondRansac.predict(line_X)

        #Calculate and print slope and intercept for the second line.
        secondSlope = secondRansac.estimator_.coef_
        secondIntercept = secondRansac.estimator_.intercept_
        # print (secondSlope)
        # print(secondIntercept)   

        #Find intercept between first line and second line
        xIntercept = (secondIntercept-firstIntercept)/(firstSlope-secondSlope)
        #xIntercept = frame.shape[1]-xIntercept

        #print (xIntercept)   

        frameRot = imutils.rotate(frame, angle)
        frameRot = imutils.resize(frameRot, width=320, height=240)
        cv.line(frameRot, (xIntercept, 0), (xIntercept, 1000), (255, 0, 0), 5)
        cv.imshow('Line Follower Original', frameRot)
        cv.line(mask, (xIntercept, 0), (xIntercept, 1000), (255, 0, 0), 5)
        cv.imshow('Line Follower Mask', mask)

        # lw = 2
        # plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
        #             label='Inliers')
        # plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
        #             label='Outliers')
        # #plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
        # plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
        #         label='RANSAC regressor')
        # plt.legend(loc='lower right')
        # plt.xlabel("Input")
        # plt.ylabel("Response")
        # plt.show()  

        # print(w)
        # princameraTarget
        cameraTarget = (w/2)-xIntercept
        print(cameraTarget)

        # arduino = 
    k = cv.waitKey(5) & 0xFF
    if k == 27 or not ret:
        if not ret:
            print('exit due to not ret')
        break