import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def get_files(filename):
    curr_dir = os.getcwd()
    files = glob.iglob(filename,
                       root_dir=curr_dir)  # This creates an iterator object containing the filenames for all the required files in the path format.
    return files


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def make_data():
    train_file_matcher = "cifar-10-batches-py/data_batch_*"
    train_file_names = get_files(train_file_matcher)
    test_file_matcher = "cifar-10-batches-py/test_batch"
    test_file_name = get_files(test_file_matcher)

    train = []
    train_labels = []
    test = []
    test_labels = []
    for i, file_name in enumerate(train_file_names):
        data_dict = unpickle(file_name)
        train.append(data_dict[b'data'])
        train_labels.append(data_dict[b'labels'])
    for j, test_filename in enumerate(test_file_name):
        data_dict = unpickle(test_filename)
        test.append(data_dict[b'data'])
        test_labels.append(data_dict[b'labels'])
    dictData = {'train_data': np.reshape(np.array(train), newshape=(
        np.array(train).shape[0] * np.array(train).shape[1], np.array(train).shape[2])),
                'train_labels': np.reshape(np.array(train_labels),
                                           newshape=(np.array(train_labels).shape[0] * np.array(train_labels).shape[
                                               1])), 'test_data': np.reshape(np.array(test), newshape=(
            np.array(test).shape[0] * np.array(test).shape[1], np.array(test).shape[2])),
                'test_labels': np.reshape(np.array(test_labels),
                                          newshape=(np.array(test_labels).shape[0] * np.array(test_labels).shape[1]))}
    return dictData


def main():
    data_dict = make_data()
    # Looking at the data:
    # print(data_dict['train_data'].shape)
    # visualizing train sample
    temp = data_dict['train_data'][21]

    # Since every row represents one example, to re-map it to image we have to form three 32,32 matrix,
    # representing RGB values

    R = temp[0:1024].reshape(32, 32)
    G = np.reshape(temp[1024:2048], newshape=(32, 32))
    B = np.reshape(temp[2048:], newshape=(32, 32))
    temp = np.dstack((R, G, B))  # for stacking all these 32,32 matrices.
    plt.imshow(temp)
    plt.show()

    # Splitting the data into train test and val sets:
    x_train, x_test, y_train, y_test = data_dict['train_data'], data_dict['test_data'], data_dict['train_labels'], \
                                       data_dict['test_labels']
    print(x_train.shape[0])
    x_train_use, y_train_use = x_train[0:49000], y_train[0:49000]
    x_val, y_val = x_train[49000:], y_train[49000:]
    print(x_train_use.shape[0])
    print(x_val.shape[0])
    print(x_test.shape[0])

    # Model Building:
    neighbours_list = [2, 3, 4, 5]
    max_acc = 0
    optimal_neighbours = 0
    for neighbours in neighbours_list:
        clf = KNeighborsClassifier(n_neighbors=neighbours)
        clf.fit(x_train_use, y_train_use)
        y_pred = clf.predict(x_val)
        acc_score = accuracy_score(y_val, y_pred)
        if acc_score > max_acc:
            max_acc = acc_score
            optimal_neighbours = neighbours
    clf = KNeighborsClassifier(n_neighbors=optimal_neighbours)
    clf.fit(x_train_use, y_train_use)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(optimal_neighbours)
    print(acc_score)


if __name__ == '__main__':
    main()
