import os
import numpy as np
import cv2
import time
import datetime

class Workplace():
    def __init__(self):
        self.WORKPLACE_FILE = "workplace_coordinates.txt"
        self.LOG_FILE = "log.txt"

        
        self.rectangle_data = np.loadtxt(self.WORKPLACE_FILE, dtype=int, delimiter=",", skiprows=1)

        self.data = []
        for item in self.rectangle_data:
            xmin, ymin, xmax, ymax, number = item
            state = 0
            row = [xmin, ymin, xmax, ymax, number, state]
            self.data.append(row)
        self.iterator = len(self.data)
        print(self.data)


    def find_active_zone(self, centroid):
        
        for i in range(self.iterator):
            if self.data[i][0] <= centroid[0] <= self.data[i][2] and self.data[i][1] <= centroid[1] <= self.data[i][3]:
                self.data[i][5] = 1
    

    def reset_state(self):
        for i in range(self.iterator):
            self.data[i][5] = 0


    def save_data(self):
        time_date = datetime.datetime.now()
        tuple_time = time_date.timetuple()
        if tuple_time[5]/59 ==1:
            intermediate_data = self.data.copy()
            intermediate_data.append(str(time_date))

            log_object  = open(self.LOG_FILE, 'a')
            log_object.write(str(intermediate_data)+'\n')
            log_object.close()
        
"""
cap = cv2.VideoCapture('left5520.jpg')
_ret, frame = cap.read()


cv2.imshow('frame', frame)

cv2.waitKey() & 0xFF == ord('q')

cap.release()
cv2.destroyAllWindows()
"""