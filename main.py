import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

path = os.getcwd()
#load the trained model
model = load_model('models/cnncat2.h5')

#initializing mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

#define some necessary vaiables
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
label = ["Open", "Closed"]
count = 0
score = 0
thick = 2
r_eye_pred=[99]
l_eye_pred=[99]

face_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Result.avi', fourcc, 20.0, (640,480))

while(True):
    ret, img = cap.read()
    out.write(img)
    height,width = img.shape[:2] 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eyes = left_eye_cascade.detectMultiScale(gray)
    right_eyes =  right_eye_cascade.detectMultiScale(gray)
   
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y) , (x+w,y+h) , (255,0,0) , 1 )

    for (x,y,w,h) in right_eyes:
            r_eye = img[y:y+h, x:x+w]
            count+=1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24,24))
            r_eye = r_eye/255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            r_eye_pred = model.predict_classes(r_eye)
            if (r_eye_pred[0] ==1):
                label = 'Open'
            if (r_eye_pred[0] ==0):
                label = 'Closed'
            break

    for (x,y,w,h) in left_eyes:
            l_eye = img[y:y+h, x:x+w]
            count+=1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24,24))
            l_eye = l_eye/255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            l_eye_pred = model.predict_classes(l_eye)
            if (l_eye_pred[0] ==1):
                label = 'Open'
            if (l_eye_pred[0] ==0):
                label = 'Closed'
            break

    if (l_eye_pred==0 and r_eye_pred == 0):
        score+=1
        cv2.putText(img,"Closed", (10, height-20),  font, 1, (255,255,255), 1, cv2.LINE_AA)
    else:
        score-=1
        cv2.putText(img, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score<0):
        score = 0
    cv2.putText(img, "Score:"+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score>15):
        cv2.imwrite(os.path.join(path,'image.jpg'), img)
        try:
            sound.play()
        except:
            pass

        if (thick < 16):
            thick = thick + 2
        else:
            thick = thick - 2
            if (thick < 2):
                thick = 2
        cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), thick)



    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()

