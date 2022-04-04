import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


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