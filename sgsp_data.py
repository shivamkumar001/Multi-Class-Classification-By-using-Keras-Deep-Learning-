#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:19:58 2018

@author: shivamkumar
"""


#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import os
import cv2

# directory

img_dir="train_1"
#/////////////////////////////
#No of classes in datasets (names)

img_classes=["gopal","pradeep","sannidhya","shivam"]

# to check image in datasets
#/////////////////////////////////////////
'''
for cl in img_classes:
    path=os.path.join(img_dir,cl)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap='gray')
        print(img_array.shape)
        plt.show()
        break
    break
'''
#///////////////////////////////////////////////////
# resize image
# ////////////////////////////////////////////////
#////////////////////////////////

img_size=80
'''
new_img_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_img_array,cmap='gray')
plt.show()
'''
#////////////////////////////////////////////////////
# now create training data
from tqdm import tqdm
training_data=[]
def create_training_data():
 
    for clas in tqdm(img_classes):
            path=os.path.join(img_dir,clas)
            #index no start from 0 1 2 3
            print(clas)
            
            class_num=img_classes.index(clas)
            print(class_num)
            
            for img in os.listdir(path):
                try:
                    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    new_array=cv2.resize(img_array,(img_size,img_size))
                    training_data.append([new_array,class_num])
                except Exception as e:
                    pass
    
       
create_training_data()
print(len(training_data))
'''
for z in training_data[:1]:  
    print(z[0])
    print("#########")
    plt.imshow(z[0],cmap="gray")

'''
#///////////////////////////////////////////////////

xs=[]
ys=[]

for features,labels in tqdm(training_data):
    xs.append(features)
    ys.append(labels)

xs=np.array(xs).reshape(-1,img_size,img_size,1)
ys=np.array(ys).reshape(-1,1)
import pickle

pickle_out=open("xs.pickle","wb")
pickle.dump(xs,pickle_out)
pickle_out.close()



pickle_out=open("ys.pickle","wb")
pickle.dump(ys,pickle_out)
pickle_out.close()
'''
pickle_in=open("xs.pickle","rb")
xs=pickle.load(pickle_in)
print(xs[0])

pickle_in_=open("ys.pickle","rb")
xs=pickle.load(pickle_in_)
print(ys[0])
'''