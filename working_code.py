#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import tensorflow as tf
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten,Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras.models import  model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json model and weights
json_file = open( "D:/codes/pycharm/EmotionRecognition/emotion_model.json",'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("D:/codes/pycharm/EmotionRecognition/emotion_model.h5")
print("Loaded model from disk")

import requests
import cv2
import numpy as np
import imutils
  

img = cv2.imread("C:/Users/meyas/OneDrive/Pictures/s2.jpg")
orig_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image

img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

# loading haarcascade_frontalface_default.xml
face_detection_model = cv2.CascadeClassifier("D:\codes\datasets\haarcascade_frontalface_default.xml")


return_faces = face_detection_model.detectMultiScale(img, scaleFactor=1.08, minNeighbors=4)  # returns a list of (x,y,w,h) tuples

# plotting the returned values
for (x, y, w, h) in return_faces:
    cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

plt.figure(figsize=(12, 12))
plt.imshow(orig_img); 

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
face_classifier = cv2.CascadeClassifier("D:\codes\datasets\haarcascade_frontalface_default.xml")

#Prediction shown in cv2
import requests
import cv2
import numpy as np
import imutils
#for IPwebcam
# url = "http://192.168.137.219:8080/shot.jpg"

while True:
#     cap = cv2.VideoCapture(url)
    cap = cv2.VideoCapture(0) # for webcam
    camera, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = emotion_model.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            print(label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




