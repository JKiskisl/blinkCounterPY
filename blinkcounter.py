import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
c=0
t0=time.time()
while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        k=1
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            k=0
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        if k==1:
            c=c+1
            print("You just blinked for the ",c , " time!")
    cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        t1 = time.time()
        break
cap.release()
total = t1-t0
print("You've blinked ",c," times, in ", total, " seconds")
cv2.destroyAllWindows()