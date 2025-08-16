import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D

(x_train,y_train),(x_test,y_test) = mnist.load_data()

grey_scale = 255

x_train = x_train.astype('float32')/grey_scale
x_test = x_test.astype('float32')/grey_scale

print("Feature matrix (x_train): ",x_train.shape)
print("Feature matrix (y_train): ",y_train.shape)
print("Feature matrix (x_test): ",x_test.shape)
print("Feature matrix (y_test): ",y_test.shape)


model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
    Dense(256,activation='relu'),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
   ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

mod = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2)
print(mod)