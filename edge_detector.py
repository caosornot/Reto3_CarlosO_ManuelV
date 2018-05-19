#Import Libraries
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import imutils

#Load image
img = cv.imread('3.jpeg')
#Convert image into HSV channel
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
angle = 180
imgHSV = imutils.rotate(imgHSV, angle)


#Define Color Boundaries
lowerBoundarie = np.array([35, 8, 0], dtype = 'uint8')
upperBoundarie = np.array([120, 255, 255], dtype = 'uint8')

#Compute mask for image
mask = cv.inRange(imgHSV, lowerBoundarie, upperBoundarie)

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
lr = linear_model.LinearRegression()
lr.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# Compare estimated coefficients
# print("Estimated coefficients (true, linear regression, RANSAC):")
# print(coef, lr.coef_, ransac.estimator_.coef_)
# print(X)


lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
#plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()


