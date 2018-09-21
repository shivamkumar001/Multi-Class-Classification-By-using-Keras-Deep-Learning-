#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:21:57 2018

@author: shivamkumar
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:49:22 2018

@author: shivamkumar
"""


#import tensorflow as tf
from tensorflow import keras
#import keras
import numpy as np
import time
import pickle
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.utils import to_categorical


#from keras.utils import normalisation
t1=time.time()
x_train=pickle.load(open("xs.pickle","rb"))
y_train=pickle.load(open("ys.pickle","rb"))
y_train=to_categorical(y_train,4)
t2=time.time()
print("time to load :",(t2-t1))

#x=x/255
# or 
x_train=x_train.astype('float32')
x_train=keras.utils.normalize(x_train)
model=Sequential()

model.add(Conv2D(128,(3,3),activation='relu',input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(activation='relu',units=128))
model.add(Dense(activation='relu',units=128))

model.add(Dense(activation='softmax',units=4))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2)


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




# Save weights to a TensorFlow Checkpoint file
model.save_weights('./my_model_sgsp3')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('my_model_sgsp3')

model.save("CNN.model_sgsp3")
