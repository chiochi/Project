from multiprocessing import Process, Value
import multiprocessing
import camera
import classify
import cv2
import time
import numpy as np

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

genderNet = cv2.dnn.readNetFromCaffe( genderProto, genderModel)

if __name__ == '__main__':
    try:
        time_face = time.time()
        faces_encoded = np.load('models/face.npy')
        known_face_names = np.load('models/name.npy')
        faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        T1 = camera.camer_read(faceCascade,genderNet)
        P1 = classify.classify_face()
        P1.faces_encoded = faces_encoded
        P1.known_face_names = known_face_names
        T1.start()
        manager = multiprocessing.Manager()
        return_name = manager.list()
        status_retrun = 0
        while 1:
            if len(T1.image) > 0 and len(T1.locations) > 0 and time.time() - time_face > 1.5:
                print( time.time() - time_face)
                return_name = manager.list()
                P = Process(target=P1.run,args=(T1.image,T1.locations,return_name,))
                P.start()
                T1.locations = []
                T1.image = []
                
                time_face = time.time()
                status_retrun = 0

            if return_name and status_retrun == 0:
                print(return_name)
                T1.face_name = return_name
                T1.save_time = 1
                status_retrun = 1
                
            if T1.exit == 1:
                break
            pass
    except KeyboardInterrupt:
        print("interrub")
    finally:
        print("Done . . !!!")