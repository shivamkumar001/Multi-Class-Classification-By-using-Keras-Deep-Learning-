#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:22:25 2018

@author: shivamkumar
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 16:14:31 2018

@author: shivamkumar
"""

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import keras
#import StringIO
import numpy as np

#import gopu_pk
#import gopu_pk_main

class_names=["gopal","pradeep","sannidhya","shivam"]

def my_img(path):
    img_size=80
    img_array=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    new_img_array=cv2.resize(img_array,(img_size,img_size))
    new_img_array=new_img_array.reshape(-1,img_size,img_size,1)
    new_img_array=keras.utils.normalize(new_img_array)
    print(new_img_array)
    return new_img_array

# eg: "1.jpg"
#print("please give image with path")
model=tf.keras.models.load_model("CNN.model_sgsp2")

#this is correct for giving input from terminal
'''
print("Please enter the Image name with(path):\n")
input_image=input()


image_=input_image

#new_img_array=[my_img(image_)]

#class_names.tags.value_counts()

prediction=model.predict(my_img(image_)) 
prediction=np.argmax(prediction,axis=1)
print(prediction)
'''
'''
prediction=model.predict([my_img(image_)]) 

np.sum(prediction[0])
np.argmax(prediction[0])

label=class_names[np.argmax(prediction[0])]

print(label)
'''
'''
#fig=plt.figure()
image = cv2.imread(image_)
print (type(image))


plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
plt.title(label)
plt.show()


print("I am 70 % confidence this is : " ,label,"\n ")



'''
import os
import numpy as np
#from PIL import image
#import pickle

#face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#mouth_cascade=cv2.CascadeClassifier('haarcascade_mouth.xml')
cap = cv2.VideoCapture(0)
from tqdm import tqdm

sampleNum=0;
while 1:
    
    ret, img = tqdm(cap.read())
    no=sampleNum
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        sampleNum+=1
        
        cv2.imwrite("/Users/shivamkumar/Documents/new_deeplearning_multiclass_classification_image/demo"+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)

        #predicted,config=recogniser.predict(gray[y:y+h,x:x+w]
    cv2.imshow("face",img)   
    print("image saved in thebdirectory")
    cv2.imshow("face",img)   
    cv2.waitKey(5)
    if(sampleNum>10):
        break


cap.release()
cv2.destroyAllWindows()

prediction=model.predict([my_img("demo1.jpg")])
np.sum(prediction[0])
np.argmax(prediction[0])

label=class_names[np.argmax(prediction[0])]

print(label)
