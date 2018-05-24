#Import Libraries
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import imutils
import math

#Load image
img = cv.imread('2.jpeg')

#Convert image into HSV channel
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
angle = 180
imgHSV = imutils.rotate(imgHSV, angle)


#Define Color Boundaries
lowerBoundarie = np.array([35, 8, 0], dtype = 'uint8')
upperBoundarie = np.array([120, 255, 255], dtype = 'uint8')

#Apply mask and obtain info of pixels
pixelInfo = cv.inRange(imgHSV, np.array(lowerBoundarie), np.array(upperBoundarie))
array = np.nonzero(pixelInfo)
transposedArray = np.transpose(array)
#print (transposedArray)

#print (array_two)
# print(array) 
# print(transposedArray)

X = transposedArray[: , 1]
y = transposedArray[: , 0]

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# print(X)
# print(Y)

# Fit line using all data
# lr = linear_model.LinearRegression()
# lr.fit(X, y)

# Robustly fit first linear model with RANSAC algorithm
firstRansac = linear_model.RANSACRegressor(residual_threshold=50)
firstRansac.fit(X, y)
firstinlier_mask = firstRansac.inlier_mask_
firstoutlier_mask = np.logical_not(firstinlier_mask)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
first_line_y_ransac = firstRansac.predict(line_X)

firstSlope = firstRansac.estimator_.coef_
firstIntercept = firstRansac.estimator_.intercept_

print (firstSlope)
print(firstIntercept)

# lw = 2
# plt.scatter(X[firstinlier_mask], y[firstinlier_mask], color='yellowgreen', marker='.',
#             label='Inliers')
# plt.scatter(X[firstoutlier_mask], y[firstoutlier_mask], color='gold', marker='.',
#             label='Outliers')
# plt.plot(line_X, first_line_y_ransac, color='cornflowerblue', linewidth=lw,
#          label='RANSAC regressor')
# plt.legend(loc='lower right')
# plt.xlabel("Input")
# plt.ylabel("Response")
# plt.show()

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

# print(len(X))
# print(len(y))

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Robustly fit Second linear model with RANSAC algorithm
secondRansac = linear_model.RANSACRegressor(residual_threshold=50)
secondRansac.fit(X, y)
secondinlier_mask = secondRansac.inlier_mask_
secondoutlier_mask = np.logical_not(secondinlier_mask)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
second_line_y_ransac = secondRansac.predict(line_X)

secondSlope = secondRansac.estimator_.coef_
secondIntercept = secondRansac.estimator_.intercept_

print (secondSlope)
print(secondIntercept)

# lw = 2
# plt.scatter(X[secondinlier_mask], y[secondinlier_mask], color='yellowgreen', marker='.',
#             label='Inliers')
# plt.scatter(X[secondoutlier_mask], y[secondoutlier_mask], color='gold', marker='.',
#             label='Outliers')
# plt.plot(line_X, first_line_y_ransac, second_line_y_ransac, color='cornflowerblue', linewidth=lw,
#          label='RANSAC regressor')
# plt.legend(loc='lower right')
# plt.xlabel("Input")
# plt.ylabel("Response")
# plt.show()

#Find intercept between first line and second line
xIntercept = (secondIntercept-firstIntercept)/(firstSlope-secondSlope)
xIntercept = img.shape[1]-xIntercept

print (xIntercept)

#Show original image with sensor
cv.line(img, (xIntercept, 0), (xIntercept, 1000), (255, 0, 0), 5)
cv.imshow('Line Detector', img)
cv.waitKey(0)



