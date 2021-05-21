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
                
                #print(0)
                
    def reset_state(self):
        for i in range(self.iterator):
            self.data[i][5] = 0

        #cv2.putText(frame, f"Active places {count_active} ", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
            
            
"""

    def draw_zone(sefl, frame, x, y):
        Rectangle_data = np.loadtxt("rectangle_coordinates.txt", dtype=int, delimiter=",")
        
        for item in Rectangle_data:
            xmin, ymin, xmax, ymax, number = item
            color = (0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(frame, str(number), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)  


cap = cv2.VideoCapture('left5520.jpg')
_ret, frame = cap.read()

centroid1 = [390, 600]
centroid2 = [705, 366]

workplace = Draw_Workplace
workplace.draw_active_zone(frame, centroid1)


cv2.circle(frame,(centroid1[0], centroid1[1]), 5, (0,0,255), -1)
cv2.circle(frame,(centroid2[0], centroid2[1]), 5, (0,0,255), -1)


cv2.imshow('frame', frame)

cv2.waitKey() & 0xFF == ord('q')

cap.release()
cv2.destroyAllWindows()
"""

"""
def draw_by_coord(frame, centroid, workplace):

    xmin, ymin, xmax, ymax = workplace

    color = (0, 0, 254)
    if xmin <= centroid[0] <= xmax and ymin <= centroid[1] <= ymax:
        color = (0, 255, 0)
    else:
        color = (0, 0, 254)
    
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)


Polinom_data = np.loadtxt("polynomial_coordinates.txt", dtype=int, delimiter=",")
#print(File_data)

Rectangle_data = np.loadtxt("rectangle_coordinates.txt", dtype=int, delimiter=",")
print(Rectangle_data)


for item in Rectangle_data:
    draw_by_coord(frame, centroid1, item)
    #xmin, ymin, xmax, ymax = item
    #print(xmin, ymin, xmax, ymax)
    #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 3)

for item in Polinom_data:
    
    poliline = item.reshape(4,2)
    #print(poliline)
    cv2.polylines(frame,[poliline],True, (0, 255, 0))


"""