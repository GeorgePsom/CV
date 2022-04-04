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
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from PIL import Image

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
print(action_categories)

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

X_train_images = tf_resize_images(X_train)

X_validation_images = tf_resize_images(X_validation)

#X_train_images = tf_resize_images(train_files)

#X_test_images = tf_resize_images(test_files)


model = Sequential()
#model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(40))

model.compile(optimizer = Adam(), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics = ['accuracy'], )
#history = model.fit(X_train_images, train_labels, batch_size = 128, epochs = 15, verbose = 2)
history = model.fit(X_train_images, Y_train, batch_size = 128, epochs = 15, verbose = 2)

#scores = model.evaluate(X_test_images, test_labels, verbose = 2)
scores = model.evaluate(X_validation_images, Y_validation, verbose = 2)

print(f'Score for fold: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')




# image_no = 3999  # change this to a number between [0, 3999] and you can see a different training image
# img = cv2.imread(f'JPEGImages/{train_files[image_no]}')
# cv2.imshow('Image', img)
# print(f'An image with the label - {train_labels[image_no]}')

print('Hello')
#cv2.waitKey(0)