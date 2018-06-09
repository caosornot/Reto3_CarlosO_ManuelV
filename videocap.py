# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:45:10 2018

@author: DELL
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(1)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('hallway-sim.mp4',fourcc, 30.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()