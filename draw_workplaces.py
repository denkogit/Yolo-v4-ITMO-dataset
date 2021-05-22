import os
import numpy as np
import cv2
import time

class Workplace():
    def __init__(self):
        self.WORKPLACE_FILE = "workplace_coordinates.txt"
        self.Rectangle_data = np.loadtxt(self.WORKPLACE_FILE, dtype=int, delimiter=",")

    
        self.data = []
        for item in self.Rectangle_data:
            xmin, ymin, xmax, ymax, number = item
            state = 0
            row = [xmin, ymin, xmax, ymax, number, state]
            self.data.append(row)
        self.iterator = len(self.data)


    def find_active_zone(self, frame, centroid):
        
        for i in range(self.iterator):
            row = xmin, ymin, xmax, ymax, number, state = self.data[i]
            #print(row)
            if xmin <= centroid[0] <= xmax and ymin <= centroid[1] <= ymax:
                self.data[i][5] = 1
             
                
    def reset_state(self):
        for i in range(self.iterator):
            self.data[i][5] = 0
      
"""
cap = cv2.VideoCapture('left5520.jpg')
_ret, frame = cap.read()


cv2.imshow('frame', frame)

cv2.waitKey() & 0xFF == ord('q')

cap.release()
cv2.destroyAllWindows()
"""