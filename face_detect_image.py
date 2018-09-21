import numpy as np
import cv2
import os
import numpy as np
#from PIL import image
#import pickle

#face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#mouth_cascade=cv2.CascadeClassifier('haarcascade_mouth.xml')
cap = cv2.VideoCapture(0)
id=input('enter user id')
from tqdm import tqdm
sampleNum=0;
while 1:
    ret, img = tqdm(cap.read())
    no=sampleNum
    print("image_no :",no)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        sampleNum+=1
        cv2.imwrite("dataSet1/shivam_img1/pshivam."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)

        #predicted,config=recogniser.predict(gray[y:y+h,x:x+w]
    cv2.imshow("face",img)   
    cv2.waitKey(5)
    if(sampleNum>10):
        break


cap.release()
cv2.destroyAllWindows()
