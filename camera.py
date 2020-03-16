import face_recognition
import threading
import requests
import _thread
import imutils
import time
import sys
import cv2
import os
from tracking.centroidtracker import CentroidTracker

ct = CentroidTracker()

MODEL_MEAN_VALUES = (124.895847746, 87.7689143744, 81.4263377603)
genderList = ['Male', 'Female']

class camer_read(threading.Thread):
    def __init__(self,faceCascade,genderNet):
        threading.Thread.__init__(self)
        self.video_capture = cv2.VideoCapture(0)
        self.faceCascade = faceCascade
        self.genderNet = genderNet
        self.save_time = 1
        self.load_model = 1
        self.face_name = []
        self.locations = []
        self.image = []
        self.exit = 0
        self.color = (0,255,0)

    def run(self):
        Start = time.time()
        start_program = 0
        while 1:
            if self.load_model == 1:
                ret, frame = self.video_capture.read()
                frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                img = frame[:,:,::-1]
                (H, W) = frame.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=6,
                    minSize=(30, 30)
                )
                count_face = 0
                locations = []
                rects = []
                for (x, y, w, h) in faces:
                    location = []
                    location.append(int(y)+5)
                    location.append(int(x+w))
                    location.append(int(y+h)+5)
                    location.append(int(x))
                    rect = tuple(location)
                    locations.append(rect)
                    rects.append([int(x),int(y)+5,int(x+w),int(y+h)+5])
                    face = frame[x:x+w,y:y+h,:]
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    self.genderNet.setInput(blob)
                    genderPreds = self.genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    if len(self.face_name) > 0 and count_face < len(self.face_name):
                        # print(count_face,self.face_name[count_face],gender)
                        cv2.putText(frame, self.face_name[count_face] + ", "+gender, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), thickness=2)
                    cv2.rectangle(frame, (x, y+5), (x+w, (y+h)+5), self.color, 2)
                    count_face += 1
                objects = ct.update(rects)
                for (objectID, centroid) in objects.items():
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                if len(locations) > 0 and self.save_time == 1:
                    self.image = img
                    self.locations = locations
                    self.save_time = 0
                
                cv2.imshow("Face Recognition",frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.exit = 1
                    break