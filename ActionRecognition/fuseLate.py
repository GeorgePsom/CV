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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Add, Input, ZeroPadding2D, AveragePooling2D, average
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from PIL import Image
from sklearn import utils


#----------------------------import data----------------------------------------------------------------




#----------------------------import models----------------------------------------------------------------


transfer_model = keras.models.load_model('data/transferModel-33')
transfer_model.trainable = False
inputTransfer = keras.Input(shape=(112,112,3))




optFlow_model = keras.models.load_model('data/optFlow-40')
optFlow_model.trainable = False
inputOptFlow = keras.Input(shape=(7, 100, 100,3))




outputs = average([transfer_model.output, optFlow_model.output])

model = Model([transfer_model.input, optFlow_model.input], outputs)

model.summary()