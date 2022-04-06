import keras.regularizers
import wget
import unzip
import cv2
import numpy as np
from helpfuncs import*
from sklearn.model_selection import StratifiedShuffleSplit
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sklearn.model_selection as sk_ms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Add, Input, ZeroPadding2D, AveragePooling2D, Conv3D, MaxPooling3D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from PIL import Image
from sklearn import utils
from moviepy.editor import *

set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
                 [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
                 [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
                 [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
                 [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
                 [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
                 [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]
action_categories = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

# test set
test_files = []

indices = [0,2,6,8,10,12,15]

for c in range(len(action_categories)):
    for i in set_1_indices[c]:
        test_files.append([f'TV-HI/test/{action_categories[c]}_{i:04d}-' + str(j + 1) + '.png' for j in indices])

test_labels = [f'{action_categories[c]}' for c in range(len(action_categories)) for i in set_1_indices[c]]


# training set
train_files = []

for c in range(len(action_categories)):
    for i in set_2_indices[c]:
        train_files.append([f'TV-HI/train/{action_categories[c]}_{i:04d}-' + str(j + 1) + '.png' for j in indices])


train_labels = [f'{action_categories[c]}' for c in range(len(action_categories)) for i in set_2_indices[c]]

print(action_categories)

#train_files, train_labels = utils.shuffle(train_files, train_labels)



categ_dict = {}
i=0

for action in action_categories:
    categ_dict[action] = i
    i += 1

for i in range(0,len(train_labels)):
    train_labels[i] = categ_dict[train_labels[i]]

for i in range(0,len(test_labels)):
    test_labels[i] = categ_dict[test_labels[i]]


IMAGE_SIZE = 100
CHANNELS = 3

def tf_resize_images(X_img_file_paths):
    X_data = []
    tf.compat.v1.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, CHANNELS))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE),
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            #print(file_path)
            img = Image.open(file_path)
            #print(np.shape(np.shape(img)))
            if len(np.shape(img))!=3:
                img2 = np.zeros((np.shape(img)[0],np.shape(img)[1],3))
                img2[:, :, 0] = img
                img2[:, :, 1] = img
                img2[:, :, 2] = img
                resized_img = sess.run(tf_img, feed_dict={X: img2})
            else:
                resized_img = sess.run(tf_img, feed_dict = {X: img})
            X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data



X_train_imagesR = []
X_test_imagesR = []

for i in range(0,len(train_files)):
    X_train_imagesR.append(tf_resize_images(train_files[i]))
    X_test_imagesR.append(tf_resize_images(test_files[i]))


X_train_images = []

for i in range(0, len(X_train_imagesR)):
    arrays = [np.array(array) for array in X_train_imagesR[i]]

    X_train_images.append(np.stack(arrays))


X_train_images = np.array(X_train_images)

X_test_images = []

for i in range(0, len(X_test_imagesR)):
    arrays = [np.array(array) for array in X_test_imagesR[i]]

    X_test_images.append(np.stack(arrays))

X_test_images = np.array(X_test_images)

print("Shape: ", X_train_images.shape)

print("Train size: ", len(X_train_images))
print("Test size: ", len(X_test_images))

#Define Model Architecture
model = Sequential()
model.add(Conv3D(16, kernel_size = (3, 3, 3), activation = 'relu',padding='same', input_shape = (7, 100, 100, 3)))
model.add(MaxPooling3D(pool_size = (2, 2, 2),padding='same'))
model.add(Dropout(0.3))
model.add(Conv3D(32, kernel_size = (3, 3, 3), activation = 'relu',padding='same'))
model.add(MaxPooling3D(pool_size = (2, 2, 2),padding='same'))
#model.add(Dropout(0.3))
model.add(Conv3D(64, kernel_size = (3, 3, 3),padding='same', activation = 'relu'))
model.add(MaxPooling3D(pool_size = (2, 2, 2),padding='same'))
model.add(Conv3D(128, kernel_size = (3, 3, 3),padding='same', activation = 'relu'))
model.add(MaxPooling3D(pool_size = (2, 2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(4))

model.compile(optimizer = Adam(), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics = ['accuracy'], )
history = model.fit(X_train_images, train_labels, batch_size = 64, epochs = 25, verbose = 2, validation_data = (np.array(X_test_images), test_labels))
#history = model.fit(X_train_images, Y_train, batch_size = 128, epochs = 15, verbose = 2, validation_data = (X_validation_images,Y_validation))

scores = model.evaluate(np.array(X_test_images), test_labels, verbose = 2)

#scores = model.evaluate(X_validation_images, Y_validation, verbose = 2)

print("Test acc: ", scores[1])
print("Test loss: ", scores[0])

#clip=VideoFileClip(f'TV-HI/tv_human_interactions_videos/{set_2[video_no]}')
#print(f'\n\nA video with the label - {set_2_label[video_no]}\n')
#clip.ipython_display(width=380)

#model.save("optFlow")