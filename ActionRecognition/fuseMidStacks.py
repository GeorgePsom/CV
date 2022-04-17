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
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Add, Input, ZeroPadding2D, AveragePooling2D, average, Maximum, Minimum, Add, Multiply, Subtract, Dot, Average
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from PIL import Image
from sklearn import utils
from clr_callback import CyclicLR

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


#----------------------------import data----------------------------------------------------------------

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

X_train_imagesR = []
X_test_imagesR = []

for i in range(0,len(train_files)):
    X_train_imagesR.append(tf_resize_images(train_files[i],IMAGE_SIZE,CHANNELS))
    X_test_imagesR.append(tf_resize_images(test_files[i],IMAGE_SIZE,CHANNELS))


X_train_imagesOpt = []

for i in range(0, len(X_train_imagesR)):
    temp_im_train = []
    for j in range(0,len(X_train_imagesR[i])):
        temp_im_train.append(rgb2gray(np.array(X_train_imagesR[i][j])))
    #temp_im_train = np.concatenate(([rgb2gray(np.array(array)) for array in X_train_imagesR[i]]), axis=1)
    temp_im_train = np.array(temp_im_train)
    temp_im_train = temp_im_train.reshape(temp_im_train.shape[1], temp_im_train.shape[2], temp_im_train.shape[0])
    X_train_imagesOpt.append(temp_im_train)


X_train_imagesOpt = np.array(X_train_imagesOpt)
print("Train size",X_train_imagesOpt.shape)

X_test_imagesOpt = []

for i in range(0, len(X_test_imagesR)):
    temp_im_test = []
    for j in range(0, len(X_test_imagesR[i])):
        temp_im_test.append(rgb2gray(np.array(X_test_imagesR[i][j])))
    temp_im_test = np.array(temp_im_test)
    #temp_im_test = np.concatenate(([rgb2gray(np.array(array)) for array in X_test_imagesR[i]]), axis=2)
    temp_im_test = temp_im_test.reshape(temp_im_test.shape[1], temp_im_test.shape[2], temp_im_test.shape[0])

    X_test_imagesOpt.append(temp_im_test)

X_test_imagesOpt = np.array(X_test_imagesOpt)
print("Test size: ", X_test_imagesOpt.shape)



test_files = [f'TV-HI/midframesTest/{action_categories[c]}_{i:04d}.png' for c in range(len(action_categories)) for i in set_1_indices[c]]
test_labels = [f'{action_categories[c]}' for c in range(len(action_categories)) for i in set_1_indices[c]]
#print(f'Set 1 to be used for test ({len(test_files)}):\n\t{test_files}')
#print(f'Set 1 labels ({len(test_labels)}):\n\t{test_labels}\n')

# training set
train_files = [f'TV-HI/midframesTrain/{action_categories[c]}_{i:04d}.png' for c in range(len(action_categories)) for i in set_2_indices[c]]
train_labels = [f'{action_categories[c]}' for c in range(len(action_categories)) for i in set_2_indices[c]]
#print(f'Set 2 to be used for train and validation ({len(train_files)}):\n\t{train_files}')
#print(f'Set 2 labels ({len(train_labels)}):\n\t{train_labels}')

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

X_train_imagesTransfer = np.array(tf_resize_images(train_files, IMAGE_SIZE, CHANNELS))

X_test_imagesTransfer = np.array(tf_resize_images(test_files, IMAGE_SIZE, CHANNELS))


#----------------------------import models----------------------------------------------------------------





transfer_model = keras.models.load_model('data/transferModel-33')
transfer_model.summary()
pretrainedTransfer = keras.Model(
    transfer_model.inputs, transfer_model.layers[-1].input, name="pretrained_transfer"
)
pretrainedTransfer.trainable = False

inputTransfer = keras.Input(shape=(112,112,3))

x = pretrainedTransfer(inputTransfer,training=False)

outputTransfer = Dense(4,activation = 'softmax')(x)

transferModel = keras.Model(inputTransfer, outputTransfer)



optFlow_model = keras.models.load_model('data/optFlow-41')
optFlow_model.summary()
pretrainedOptFlow = keras.Model(
    optFlow_model.inputs, optFlow_model.layers[-1].input, name="pretrained_optFlow"
)
pretrainedOptFlow.trainable = False

inputOptFlow = keras.Input(shape=(112, 112, 8))
#inputOptFlow = keras.Input(shape=(7, 100, 100, 3))

x = pretrainedOptFlow(inputOptFlow,training=False)

outputOptFlow = Dense(4,activation = 'softmax')(x)

optFlowModel = keras.Model(inputOptFlow, outputOptFlow)



#outputs = average([transferModel.output, optFlowModel.output])
#outputs = Maximum()([transferModel.output, optFlowModel.output])
#outputs = Minimum()([transferModel.output, optFlowModel.output])
#outputs = Add()([transferModel.output, optFlowModel.output])
#outputs = Multiply()([transferModel.output, optFlowModel.output])
#outputs = Subtract()([transferModel.output, optFlowModel.output])
#outputs = Average()([transferModel.output, optFlowModel.output])
#outputs1 = Maximum()([transferModel.output, optFlowModel.output])
#outputs2 = Minimum()([transferModel.output, optFlowModel.output])
#outputs = Subtract()([outputs1, outputs2])

outputs = Maximum()([transferModel.output, optFlowModel.output])

model = Model([transferModel.input, optFlowModel.input], outputs)

model.summary()

x = model([transferModel.input, optFlowModel.input],training=False)
output_new = Dense(32)(x)
output_new = Dense(64)(x)
output_new = Dense(4)(x)
new_model = keras.Model([transferModel.input, optFlowModel.input],output_new)



#MIN_LR = 1e-4
#MAX_LR = 1e-1
MIN_LR = 1e-7
MAX_LR = 1e-2
BATCH_SIZE = 1
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 30

clr = CyclicLR(
	mode=CLR_METHOD,
	base_lr=MIN_LR,
	max_lr=MAX_LR,
	step_size= STEP_SIZE * (X_train_imagesTransfer.shape[0] // BATCH_SIZE))



new_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), optimizer=Adam(learning_rate=0.001),
                       metrics=['acc'])

def lrdecay(epoch):
    lr = 0.005
    if epoch > 26:
        lr *= 0.0625
    elif epoch > 22:
        lr *= 0.125
    elif epoch > 18:
        lr *= 0.25
    elif epoch > 13:
        lr *= 0.5
    #print('Learning rate: ', lr)
    return lr
  # if epoch < 40:
  #   return 0.01
  # else:
  #   return 0.01 * np.math.exp(0.03 * (40 - epoch))
lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay

#history = new_model.fit([X_train_imagesTransfer, X_train_imagesOpt], train_labels, batch_size = 1, epochs = 30, verbose = 2, callbacks=[clr], validation_data = ([X_test_imagesTransfer, X_test_imagesOpt], test_labels))
history = new_model.fit([X_train_imagesTransfer, X_train_imagesOpt], train_labels, batch_size = 1, epochs = 50, verbose = 2, validation_data = ([X_test_imagesTransfer, X_test_imagesOpt], test_labels))


scores = new_model.evaluate([X_test_imagesTransfer, X_test_imagesOpt], test_labels, verbose = 2)

print(f'Score for fold: {new_model.metrics_names[0]} of {scores[0]}; {new_model.metrics_names[1]} of {scores[1]*100}%')