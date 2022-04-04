import wget
import unzip
import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

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

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
X_train = []
Y_train  = []
X_validation  = []
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
print("TRAIN: ",  Y_train)
print("VALIDATION: ",  Y_validation)


 # y_train, y_test = train_labels[train_index], train_labels[test_index]

# image_no = 3999  # change this to a number between [0, 3999] and you can see a different training image
# img = cv2.imread(f'JPEGImages/{train_files[image_no]}')
# cv2.imshow('Image', img)
# print(f'An image with the label - {train_labels[image_no]}')

print('Hello')
cv2.waitKey(0)