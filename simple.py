#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:50:37 2018

@author: arpit-mint
"""

#Part-1 : Building a CNN
#Import the Keras libraries
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initializing the CNN
classifier=Sequential()

#Step 1: Convolution
classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(64,64,3), activation='relu'))
#Step 2: Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#2nd layer
classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#3rd layer
classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#Step 3: Flattening
classifier.add(Flatten())

#Step 4: Full Connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(rate=0.5))
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units=1,activation='sigmoid'))#overfitting

#sgd=optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
#activation='softmax'
classifier.compile(optimizer=SGD(lr=0.03, momentum=0.9, nesterov=False), loss='binary_crossentropy', metrics=['accuracy'])
#Compiling the CNN
#classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#Part 2: Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

classifier.save('model.h5')
#Single Image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/dogs/dog.1.jpg', target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)

result=classifier.predict(test_image)
#training_set.class_indices
if result[0][0]==1:
    print('Dog')
else:
    print('Cat')    
