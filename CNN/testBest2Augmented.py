from __future__ import print_function
import tensorflow as tf
import keras
import sklearn.model_selection as sk_ms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



#---------------------------------------------------------------------------Adaptive Learning-----------------------------------------------------------



fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

print(len(train_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0

def scheduler(epoch, lr):
    if (epoch-5)>=0 and epoch % 5 == 0:
        #print("Epoch: "+ str(epoch) + " lr: " + str(lr))
        return lr/2
    else:
        return lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# create data generator
datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05, zoom_range = 0.1)
# (rotation_range = 10, horizontal_flip = True, zoom_range = 0.1)
# rotation_range = 10, fill_mode = 'nearest'

train_imagesAug = train_images.reshape(train_images.shape[0], 28, 28, 1)

# create iterator
it = datagen.flow(train_imagesAug, train_labels)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10))

model.compile(optimizer = Adam(learning_rate = 1e-3), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics = ['accuracy'])

history = model.fit(it, batch_size = 128, epochs = 15, verbose = 2, callbacks=[callback])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("Test accuracy: "+str(test_acc));
print("Test loss: "+str(test_loss));


#---------------------------------------------------------------------------Kernel size-----------------------------------------------------------



fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

print(len(train_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0

# create data generator
datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05, zoom_range = 0.1)
# rotation_range = 10, fill_mode = 'nearest'

train_imagesAug = train_images.reshape(train_images.shape[0], 28, 28, 1)

# create iterator
it = datagen.flow(train_imagesAug, train_labels)

#Define Model Architecture
model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size = (5, 5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10))

model.compile(optimizer = Adam(), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics = ['accuracy'], )

history = model.fit(it, batch_size = 128, epochs = 15, verbose = 2, callbacks=[callback])


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("Test accuracy: "+str(test_acc));
print("Test loss: "+str(test_loss));







