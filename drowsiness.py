import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
import os
from pygame import mixer
import time
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

mixer.init()
sound = mixer.Sound('alarm.wav')

data=[]
labels=[]
classes=2

for i in range(classes):
    path=os.path.join("train dizziness",str(i))
    images=os.listdir(path)
    for a in images:
        image=load_img((os.path.join(path,a)),target_size=(30,30))
        gray_image = ImageOps.grayscale(image)
        image=img_to_array(gray_image)
        data.append(image)
        labels.append(i)

data=np.array(data)
labels=np.array(labels)

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=2)

y_train=to_categorical(y_train,2)
y_test=to_categorical(y_test,2)

model=Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(30,30,1)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPool2D(3,3))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(2,activation="softmax"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=8,validation_data=(x_test,y_test))

face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
leye = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_righteye_2splits.xml')

lbl=['Close','Open']
score=0

while cap.isOpened():
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects=face.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),5)
        roi_face=gray[y:y+h,x:x+w]
            
        left_rects=leye.detectMultiScale(roi_face,1.5,7)
        for (lx,ly,lw,lh) in left_rects:
            l_eye=roi_face[ly:ly+lh,lx:lx+lw]
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255.0
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = np.argmax(model.predict(l_eye))
            break
                
        right_rects=reye.detectMultiScale(roi_face,1.5,7)
        for (rx,ry,rw,rh) in right_rects:
            r_eye=roi_face[ry:ry+rh,rx:rx+rw]
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255.0
            r_eye=r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = np.argmax(model.predict(r_eye))
            break
                
        if(lbl[rpred]=="Close" and lbl[lpred]=="Close"):
            score+=1
            cv2.putText(frame,"Closed",(50,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),6,cv2.LINE_AA)
                
        else:
            score=0
            cv2.putText(frame,"Open",(50,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),6,cv2.LINE_AA)
                
        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(170,170),cv2.FONT_HERSHEY_SIMPLEX, 6,(255,255,255),1,cv2.LINE_AA)
            
        if score>15:
            try:
                sound.play()
            except:
                pass
            
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()