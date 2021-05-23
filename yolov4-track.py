import os
import numpy as np
import cv2
import time

from motpy import Detection, MultiObjectTracker, NpImage, Box
from motpy.core import setup_logger
from motpy.testing_viz import draw_detection, draw_track, draw_centre
from draw_workplaces import Workplace

logger = setup_logger(__name__, 'DEBUG', is_main=True)

VIDEO_PATH = "14_08.mp4"
WEIGHTS_PATH = "assets/custom-yolov4-tiny-detector_final.weights"
CONFIG_PATH = "assets/custom-yolov4-tiny-detector.cfg"

class RunDetection():
    def __init__(self):
        self.net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #must be enabled for CPU
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #must be enabled for CPU
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/float(255.0), swapRB=True) #float is important for Python version 2!!!  
    
        self.CONFIDENCE_THRESHOLD = 0.7 
        self.NMS_THRESHOLD = 0.4

    def process_image(self, image):

        classes, scores, boxes = self.model.detect(image, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
    
        out_detections = []
        for (classid, score, box) in zip(classes, scores, boxes):
            confidece_score = np.float32(score[0])

            x, y, w, h = box
            xmin = int(x)
            ymin = int(y)
            xmax = int(x + w)
            ymax = int(y +h)
            out_detections.append(Detection(box=[xmin, ymin, xmax, ymax], score=confidece_score))
        return out_detections

def run():
    # prepare multi object tracker
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}

    dt = 1 / 15.0  # assume 15 fps

    tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)
    detector = RunDetection()
    workplace = Workplace()

    # open camera
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 24400)

    # used to record the time at which we processe current frame
    prev_frame_time = 0; new_frame_time = 0;
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # get FPS   
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = round(fps, 2)

        #frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

        # run detector on current frame
        detections = detector.process_image(frame)
        #logger.debug(f'detections: {detections}')

        # get taracks 
        tracker.step(detections)
        tracks = tracker.active_tracks(min_steps_alive=3)
        #logger.debug(f'tracks: {tracks}')

        #reset state of workplaces 1-active, 0-not active
        workplace.reset_state()

        # draw detection 
        for det in detections:
            draw_detection(frame, det) 
        # draw tracks
        for track in tracks:
            draw_track(frame, track)
            draw_centre(frame, track.centroid)
            workplace.find_active_zone(track.centroid)

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
    
        #save workplaces state to log.txt file 
        workplace.save_data()

        cv2.putText(frame, f"Total amount of places: {len(workplace.data)} ", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
        cv2.putText(frame, f"Active places: {active_places} ", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
        cv2.putText(frame, f"FPS: {fps}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
        cv2.putText(frame, f"Detected people: {len(tracks)}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)

        cv2.imshow('frame', frame)

        # stop demo by pressing 'q'
        if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()