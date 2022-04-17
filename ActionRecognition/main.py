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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# url1 = 'http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip'
#
# url2 = 'http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip'
# filename1 = wget.download(url1)
# filename2 = wget.download(url2)
#
# filename1
# filename2

withAugmentation = True;

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
print(action_categories)

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


X_train, Y_train, X_validation, Y_validation = stratify_data(train_files, train_labels)

stratification_check(Y_train, Y_validation)


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
            img = Image.open('JPEGImages/'+file_path)
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

#X_train_imagessss = tf_resize_images(X_train)

X_validation_images = np.array(tf_resize_images(X_validation))

#X_train_images = np.array(tf_resize_images(X_train))
X_train_images = tf_resize_images(X_train)

X_test_images = np.array(tf_resize_images(test_files))

if withAugmentation == True:
    #datagen = ImageDataGenerator(height_shift_range=0.1, width_shift_range=0.1, rotation_range = 0.1, zoom_range = 0.1, fill_mode = 'nearest')
    datagen = ImageDataGenerator(height_shift_range=0.1, width_shift_range=0.1, zoom_range=0.02, fill_mode='nearest', rotation_range = 0.05)
    #22.487346827983856%
    #22.668112814426422%

    #datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range = 0.1)
    # (rotation_range = 10, horizontal_flip = True, zoom_range = 0.1)
    # rotation_range = 10, fill_mode = 'nearest'

    train_imagesAug = X_train_images.reshape(X_train_images.shape[0], 112, 112, 3)
    # create iterator
    it = datagen.flow(train_imagesAug, Y_train)


def res_identity(x, filters):


  x_skip = x
  f1, f2 = filters


  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)


  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)


  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(x)
  x = BatchNormalization()(x)



  x = Add()([x, x_skip])
  x = Activation(relu)(x)

  return x




def res_conv(x, s, filters):

  x_skip = x
  f1, f2 = filters


  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(x)

  x = BatchNormalization()(x)
  x = Activation(relu)(x)


  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)


  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(x)
  x = BatchNormalization()(x)


  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.01))(x_skip)
  x_skip = BatchNormalization()(x_skip)


  x = Add()([x, x_skip])
  x = Activation(relu)(x)

  return x



def resnet(train_im):

  input_im = Input(shape=(train_im.shape[1], train_im.shape[2], train_im.shape[3]))
  x = ZeroPadding2D(padding=(3, 3))(input_im)



  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)



  x = res_conv(x, s=1, filters=(64, 128))
  x = res_identity(x, filters=(64, 128))
  x = res_identity(x, filters=(64, 128))



  x = AveragePooling2D((2, 2), padding='same')(x)

  x = Flatten()(x)
  x = Dense(256, activation= 'relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
  x = Dense(512, activation= 'relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
  x = Dense(40, kernel_initializer='he_normal')(x) #multi-class

  # define the model

  model = tf.keras.Model(inputs=input_im, outputs=x, name='Resnet50')

  return model



resnet_model = resnet(X_train_images)

resnet_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), optimizer=Adam(learning_rate=1e-3),
                       metrics=['acc'])
resnet_model.summary()

def lrdecay(epoch):
    lr = 1e-4
    if epoch > 10:
        lr *= 0.25
    elif epoch > 5:
        lr *= 0.5
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    return lr

lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay



if withAugmentation == True:
    history = resnet_model.fit(it, epochs = 30, verbose = 1, callbacks=[lrdecay])
else:
    history = resnet_model.fit(X_train_images, Y_train, epochs=30, verbose=1, callbacks=[lrdecay])


scores = resnet_model.evaluate(X_test_images, test_labels, verbose = 2)

print(f'Score for fold: {resnet_model.metrics_names[0]} of {scores[0]}; {resnet_model.metrics_names[1]} of {scores[1]*100}%')




# image_no = 3999  # change this to a number between [0, 3999] and you can see a different training image
# img = cv2.imread(f'JPEGImages/{train_files[image_no]}')
# cv2.imshow('Image', img)
# print(f'An image with the label - {train_labels[image_no]}')

#cv2.waitKey(0)