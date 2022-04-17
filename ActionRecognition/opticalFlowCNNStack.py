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

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

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

indices = [1,3,6,8,9,11,13,15]
#indices = [6,9,13]

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


IMAGE_SIZE = 112
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
    temp_im_train = []
    for j in range(0,len(X_train_imagesR[i])):
        temp_im_train.append(rgb2gray(np.array(X_train_imagesR[i][j])))
    #temp_im_train = np.concatenate(([rgb2gray(np.array(array)) for array in X_train_imagesR[i]]), axis=1)
    temp_im_train = np.array(temp_im_train)
    temp_im_train = temp_im_train.reshape(temp_im_train.shape[1], temp_im_train.shape[2], temp_im_train.shape[0])
    X_train_images.append(temp_im_train)


X_train_images = np.array(X_train_images)
print("Train size",X_train_images.shape)

X_test_images = []

for i in range(0, len(X_test_imagesR)):
    temp_im_test = []
    for j in range(0, len(X_test_imagesR[i])):
        temp_im_test.append(rgb2gray(np.array(X_test_imagesR[i][j])))
    temp_im_test = np.array(temp_im_test)
    #temp_im_test = np.concatenate(([rgb2gray(np.array(array)) for array in X_test_imagesR[i]]), axis=2)
    temp_im_test = temp_im_test.reshape(temp_im_test.shape[1], temp_im_test.shape[2], temp_im_test.shape[0])

    X_test_images.append(temp_im_test)

X_test_images = np.array(X_test_images)
print("Test size: ", X_test_images.shape)

#Define Model Architecture
model = Sequential()
#model.add(Conv2D(64, kernel_size = (5, 5), activation = 'relu',padding='same', input_shape = (112, 112, len(indices))))
model.add(Conv2D(64, kernel_size = (15, 15), activation = 'relu',padding='same', input_shape = (112, 112, len(indices))))
model.add(MaxPooling2D(pool_size = (2, 2),padding='same'))
model.add(Dropout(0.5))
#model.add(Conv2D(112, kernel_size = (5, 5), activation = 'relu',padding='same'))
model.add(Conv2D(112, kernel_size = (10, 10), activation = 'relu',padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2),padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(4, activation = 'softmax'))

model.compile(optimizer = Adam(learning_rate = 0.0001), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics = ['accuracy'], )
history = model.fit(X_train_images, train_labels, batch_size = 10, epochs = 20, verbose = 2, validation_data = (np.array(X_test_images), test_labels))
#history = model.fit(X_train_images, Y_train, batch_size = 128, epochs = 15, verbose = 2, validation_data = (X_validation_images,Y_validation))

scores = model.evaluate(np.array(X_test_images), test_labels, verbose = 2)

#scores = model.evaluate(X_validation_images, Y_validation, verbose = 2)

print("Test acc: ", scores[1])
print("Test loss: ", scores[0])

#clip=VideoFileClip(f'TV-HI/tv_human_interactions_videos/{set_2[video_no]}')
#print(f'\n\nA video with the label - {set_2_label[video_no]}\n')
#clip.ipython_display(width=380)

model.save("optFlow")