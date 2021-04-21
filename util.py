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
    def fit_transfrom(self, data):
        self.min = np.min(data)
        self.max = np.max(data)
        return (data - self.min) / (self.max - self.min)
    def transfrom(self, data):
        return (data - self.min) / (self.max - self.min)
    def fit(self, data):
        self.min = np.min(data)
        self.max = np.max(data)

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
