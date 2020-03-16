import face_recognition as fr
import numpy as np
import os
import cv2
encoded_face = []
encoded_name = []
for dirpath, dnames, fnames in os.walk("./dataset"):
    d = dirpath.split("./dataset\\")
    for df in d:
        if df != '' and df != './dataset':
            for f in fnames:
                # encoding.append(df)
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file("dataset/"+df+"/" + f)
                    face_locations = fr.face_locations(face, number_of_times_to_upsample=0, model="cnn")
                    if face_locations:
                        image = face[face_locations[0][0]:face_locations[0][2],face_locations[0][3]:face_locations[0][1],:]
                        image = cv2.resize(image,(50,50))
                        train = fr.face_encodings(image)
                        if train is not None:
                            if len(train) > 0:
                                print("dataset/"+df+"/" + f,f,df)
                                encoded_face.append(train[0])
                                encoded_name.append(df)
                    # print(encoding)

np.save('models/face.npy', encoded_face)   
np.save('models/name.npy', encoded_name)   


