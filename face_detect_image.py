# import all dependencies
import numpy as np
import cv2
import os
import numpy as np

# .xml file of frontal face
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#mouth_cascade=cv2.CascadeClassifier('haarcascade_mouth.xml')

# 0 for primary camera 1 for secondary camera
cap = cv2.VideoCapture(0)

# use for showing contineous showing progress of images
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
    # it will take 50000  pictures of face
    if(sampleNum>50000):
        break
# important 
# release camera 
cap.release()
# destroy all open windows
cv2.destroyAllWindows()
