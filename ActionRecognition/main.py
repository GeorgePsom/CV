import wget
import unzip
#import cv2
import numpy as np
from helpfuncs import*
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import sklearn.model_selection as sk_ms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# url1 = 'http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip'
#
# url2 = 'http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip'
# filename1 = wget.download(url1)
# filename2 = wget.download(url2)
#
# filename1
# filename2

with open('ImageSplits/train.txt', 'r') as f:
    train_files = list(map(str.strip, f.readlines()))
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    #print(f'Train files ({len(train_files)}):\n\t{train_files}')
    #print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

with open('ImageSplits/test.txt', 'r') as f:
    test_files = list(map(str.strip, f.readlines()))
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
    #print(f'Test files ({len(test_files)}):\n\t{test_files}')
    #print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')

action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
#print(f'Action categories ({len(action_categories)}):\n{action_categories}')

X_train, Y_train, X_validation, Y_validation = stratify_data(train_files, train_labels)

stratification_check(Y_train, Y_validation)


model = Sequential()
#model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10))

model.compile(optimizer = Adam(), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics = ['accuracy'], )
history = model.fit(X_train, Y_train, batch_size = 128, epochs = 15, verbose = 2)

#scores = model.evaluate(test_files, test_labels, verbose = 2)





# image_no = 3999  # change this to a number between [0, 3999] and you can see a different training image
# img = cv2.imread(f'JPEGImages/{train_files[image_no]}')
# cv2.imshow('Image', img)
# print(f'An image with the label - {train_labels[image_no]}')

print('Hello')
#cv2.waitKey(0)