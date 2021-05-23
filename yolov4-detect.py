import cv2
import time
from draw_workplaces import Workplace
import threading

CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
#with open("classes.txt", "r") as f:
with open("assets/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

frame_set_no = 880
cap = cv2.VideoCapture("14_08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_set_no)

net = cv2.dnn.readNet("assets/custom-yolov4-tiny-detector_final.weights", "assets/custom-yolov4-tiny-detector.cfg")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #must be enabled for GPU
### Either DNN_TARGET_CUDA_FP16 or DNN_TARGET_CUDA must be enabled for GPU
### DNN_TARGET_CUDA shows better perf. (default for most CNN)
### DNN_TARGET_CUDA_FP16 shows faster, but only supported for recent GPUs
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16) 
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #must be enabled for CPU
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #must be enabled for CPU
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE) #OpenVINO
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #must be enabled for CPU

model = cv2.dnn_DetectionModel(net)
#model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
model.setInputParams(size=(416, 416), scale=1/float(255.0), swapRB=True) #float is important for Python version 2!!!

workplace = Workplace()

# used to record the time at which we processe current frame
prev_frame_time = 0; new_frame_time = 0;
while cv2.waitKey(1) < 1:
    (grabbed, frame) = cap.read()
    if not grabbed:
        break
    # get FPS   
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = round(fps, 2)

    # run detection
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    #reset state of workplaces 1-active, 0-not active
    workplace.reset_state()

    for (classid, score, box) in zip(classes, scores, boxes):
        x, y, w, h = box
        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w)
        ymax = int(y +h)

        xcentre = int((xmin + xmax) / 2)
        ycentre = int((ymin + ymax) / 2)

        scoreVal = score[0]*100
        scoreVal = round(scoreVal, 3)
        
        centroid = [xcentre, ycentre]
        cv2.circle(frame, (xcentre, ycentre), 3, (0,0,255), -1)

        workplace.find_active_zone(centroid)
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid[0]]}: {str(scoreVal)}%"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #save workplaces state to log.txt file 
    workplace.save_data()

    #draw workplace boxes 
    active_places = 0
    for item in workplace.data:
        xmin, ymin, xmax, ymax, number, state = item
        if state == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
            active_places+=1
        cv2.putText(frame, str(number), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

   
    cv2.putText(frame, f"Total amount of places: {len(workplace.data)} ", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
    cv2.putText(frame, f"Active places: {active_places} ", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
    cv2.putText(frame, f"FPS: {fps}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
    cv2.putText(frame, f"Detected people: {len(boxes)}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
    cv2.imshow("detections", frame)

cap.release()
cv2.destroyAllWindows()

