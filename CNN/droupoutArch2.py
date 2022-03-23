import tensorflow as tf
import sklearn.model_selection as sk_ms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

print(len(train_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0
acc_per_fold = []
loss_per_fold = []

kfold = sk_ms.KFold(n_splits = 5, shuffle = True)

fold_no = 1

for train, test in kfold.split(train_images, train_labels):
    #Define Model Architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(10))

    model.compile(optimizer = Adam(), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics = ['accuracy'], )
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    history = model.fit(train_images[train], train_labels[train], batch_size = 128, epochs = 15, verbose = 2, validation_data = (train_images[test],train_labels[test]))

    scores = model.evaluate(train_images[test], train_labels[test], verbose = 2)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])


    if fold_no == 1:
        xaxis = list(range(1, 15 + 1))
        plt.plot(xaxis, history.history['accuracy'], label='accuracy')
        plt.plot(xaxis, history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig('data/plots/Dropout2Acc.png')
        plt.close()

        plt.plot(xaxis, history.history['loss'], label='loss')
        plt.plot(xaxis, history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig('data/plots/Dropout2Loss.png')
        plt.close()

        model.save_weights('data/model/Dropout2weights')
        #model.save('data/model/model' + str(fold_no))

        model_json = model.to_json()
        with open('data/architecture/architectureDropout2.json', "w") as json_file:
            json_file.write(model_json)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
    print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

