#Import Libraries
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

#Load image
img = cv.imread('3.jpeg')
#Convert image into HSV channel
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#Define Color Boundaries
lowerBoundarie = np.array([35, 8, 0], dtype = 'uint8')
upperBoundarie = np.array([120, 255, 255], dtype = 'uint8')

#Compute mask for image
mask = cv.inRange(imgHSV, lowerBoundarie, upperBoundarie)

edges = cv.Canny(mask, 400, 500)
edges_two = cv.Canny(mask, 0, 1000)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(mask,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges_two,cmap = 'gray')
# plt.title('Edge_two Image'), plt.xticks([]), plt.yticks([])

plt.show()