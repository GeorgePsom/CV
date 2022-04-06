import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image


def stratify_data(train_files, train_labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    X_train = []
    Y_train = []
    X_validation = []
    Y_validation = []
    for train_index, test_index in sss.split(train_files, train_labels):
        for index in train_index:
            X_train = np.append(X_train, train_files[index])
            Y_train = np.append(Y_train, train_labels[index])

        for index in test_index:
            X_validation = np.append(X_validation, train_files[index])
            Y_validation = np.append(Y_validation, train_labels[index])

    print(Y_train.size)
    print(Y_validation.size)

    return X_train, Y_train, X_validation, Y_validation

def stratification_check(Y_train, Y_validation):
    all_labels = np.unique(Y_train)
    label_counts = {}

    for label in all_labels:
        label_counts[label] = [0, 0]

    for lab_train in Y_train:
        label_counts[lab_train][0] += 1

    for lab_validate in Y_validation:
        label_counts[lab_validate][1] += 1

    print("Label counts: ", label_counts)

    for label in label_counts.keys():
        label_counts[label][0] /= Y_train.size
        label_counts[label][1] /= Y_validation.size

    print("Labels avg: ", label_counts)

def tf_resize_images(X_img_file_paths, image_size, channels):
    X_data = []
    tf.compat.v1.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, channels))
    tf_img = tf.image.resize_images(X, (image_size, image_size),
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