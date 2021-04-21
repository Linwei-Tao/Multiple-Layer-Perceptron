import numpy as np


def oneHotEncoding(data, n_class):
    onehot = np.zeros((data.shape[0], n_class))
    for i in range(data.shape[0]):
        k = int(data[i])
        onehot[i, k] = int(1)
    return onehot.squeeze()  # len(y) * 10 features


def softmax(data):
    output = []
    for i in range(0, data.shape[0]):
        equ = np.exp(data[i]) / np.exp(data[i]).sum()
        output.append(equ)
    return np.vstack(output)


class MinMax_transformer:
    def fit_transform(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return (data - self.min) / (self.max - self.min)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return self


class Standardisation(object):
    def fit_transform(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)

    def transform(self, data):
        return (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return self


def crossValidation(X, y, train_test_split, epoch, n_fold=5):
    n_data = X.shape[0]
    fold_index = epoch % n_fold
    val_rate = 1 - train_test_split
    split = round(fold_index * val_rate * 100) / 100
    index = [i for i in range(n_data)]
    val_index = index[int(n_data * split):int(n_data * (split + val_rate))]
    train_index = index[:int(n_data * split)]
    train_second_half = index[:int(n_data * (split + val_rate))]
    train_index.extend(train_second_half)
    X_train = np.array(X[train_index])
    y_train = np.array(y[train_index])
    X_val = np.array(X[val_index])
    y_val = np.array(y[val_index])
    return X_train, y_train, X_val, y_val
