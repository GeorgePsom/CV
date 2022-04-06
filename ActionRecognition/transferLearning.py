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
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Add, Input, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from PIL import Image
from sklearn import utils

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
test_files = [f'TV-HI/midframesTest/{action_categories[c]}_{i:04d}.png' for c in range(len(action_categories)) for i in set_1_indices[c]]
test_labels = [f'{action_categories[c]}' for c in range(len(action_categories)) for i in set_1_indices[c]]
#print(f'Set 1 to be used for test ({len(test_files)}):\n\t{test_files}')
#print(f'Set 1 labels ({len(test_labels)}):\n\t{test_labels}\n')

# training set
train_files = [f'TV-HI/midframesTrain/{action_categories[c]}_{i:04d}.png' for c in range(len(action_categories)) for i in set_2_indices[c]]
train_labels = [f'{action_categories[c]}' for c in range(len(action_categories)) for i in set_2_indices[c]]
#print(f'Set 2 to be used for train and validation ({len(train_files)}):\n\t{train_files}')
#print(f'Set 2 labels ({len(train_labels)}):\n\t{train_labels}')

train_files, train_labels = utils.shuffle(train_files, train_labels)

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

X_train_images = tf_resize_images(train_files, IMAGE_SIZE, CHANNELS)

X_test_images = tf_resize_images(test_files, IMAGE_SIZE, CHANNELS)


base_model = keras.models.load_model('data/baseModel')
base_model.summary()
pretrained = keras.Model(
    base_model.inputs, base_model.layers[-1].input, name="pretrained_model"
)
pretrained.trainable = False

#base_model.summary()
input = keras.Input(shape=(112,112,3))
#x = base_model.layers[-1].output(input)
x = pretrained(input,training=False)
#x = Dense(16,kernel_regularizer=keras.regularizers.l2(0.01), activation = 'relu')(x)
#x = Dense(8,kernel_regularizer=keras.regularizers.l2(0.01), activation = 'relu')(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Dense(64, activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Dense(16, activation = 'relu')(x)
x = Dropout(0.25)(x)
#x = base_model

output = Dense(4,activation = 'softmax')(x)
new_model = keras.Model(input, output)
new_model.summary()

new_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), optimizer=Adam(learning_rate=1e-4),
                       metrics=['acc'])


def lrdecay(epoch):
    lr = 1e-4
    if epoch > 20:
        lr *= 0.0625
    elif epoch > 15:
        lr *= 0.125
    elif epoch > 10:
        lr *= 0.25
    elif epoch > 5:
        lr *= 0.5
    #print('Learning rate: ', lr)
    return lr
  # if epoch < 40:
  #   return 0.01
  # else:
  #   return 0.01 * np.math.exp(0.03 * (40 - epoch))
lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay



history = new_model.fit(np.array(X_train_images), train_labels, batch_size = 1, epochs = 25, verbose = 1, callbacks=[lrdecay])
#history = model.fit(X_train_images, Y_train, batch_size = 128, epochs = 15, verbose = 2)

scores = new_model.evaluate(np.array(X_test_images), test_labels, verbose = 2)
#scores = model.evaluate(X_validation_images, Y_validation, verbose = 2)

print(f'Score for fold: {new_model.metrics_names[0]} of {scores[0]}; {new_model.metrics_names[1]} of {scores[1]*100}%')

#new_model.save("transferModel")

