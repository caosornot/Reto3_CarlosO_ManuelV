#Import Libraries
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

#Load image
img = cv.imread('3.jpeg')
#Convert image into HSV channel
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#Define Color Boundaries
lowerBoundarie = np.array([35, 8, 0], dtype = 'uint8')
upperBoundarie = np.array([120, 255, 255], dtype = 'uint8')

#Compute mask for image
mask = cv.inRange(imgHSV, lowerBoundarie, upperBoundarie)

pixelInfo = cv.inRange(imgHSV, np.array(lowerBoundarie), np.array(upperBoundarie))
#array_two = cv.inRange(mask, 0, 1)
array = np.nonzero(pixelInfo)
transposedArray = np.transpose(array)
#print (transposedArray)

#print (array_two)
print(type(array)) 
print(type(transposedArray))

x = transposedArray[0]
hola = x[0]
print (x)
print (hola)

# ransac = linear_model.RANSACRegressor()
# ransac.fit (X,y)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# # Predict data of estimated models
# line_X = np.arange(X.min(), X.max())[:, np.newaxis]

# line_y_ransac = ransac.predict(line_X)
# # Compare estimated coefficients
# print("Estimated coefficients (RANSAC):")
# print( ransac.estimator_.coef_)


