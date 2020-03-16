import face_recognition as fr
import multiprocessing
import requests
import _thread
import imutils
import numpy as np
import time
import sys
import cv2
import os

class classify_face:
    def __init__(self):
        self.faces_encoded = []
        self.known_face_names = []
        self.face_names = []
        self.distances = []

    def run(self,image,locations,return_name):
        face_names = []
        unknown_face_encodings = fr.face_encodings(image, locations)
        for face_encoding in unknown_face_encodings:
            matches = fr.compare_faces(self.faces_encoded, face_encoding)
            name = "Unknown"
            face_distances = fr.face_distance(self.faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.20:
                name = self.known_face_names[best_match_index]
            return_name.append(name)
            rock_face = 0
            print(name, matches[best_match_index] , face_distances[best_match_index])
        return