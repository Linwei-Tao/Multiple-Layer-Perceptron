import numpy as np
import nn
import pickle
import os
import errno
from datetime import datetime
import os
import util
import math
import matplotlib.pyplot as plt
import time
import pandas as pd


class MLP:
    def __init__(self, n_neurons=[128, 256, 10], activation=[None, 'relu', 'relu'], dropout=0.2, bn=True):
        self.bn = bn
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.activation = activation
        self.mode = 'train'
        self.layers = []
        for i in range(len(n_neurons) - 1):
            self.layers.append(nn.Linear(n_in=n_neurons[i],
                                         n_out=n_neurons[i + 1],
                                         activation=self.activation[i + 1],
                                         activation_last_layer=self.activation[i],
                                         layer_index=i,
                                         bn=self.bn,
                                         dropout=self.dropout,
                                         output_layer=(i == len(n_neurons) - 2)))

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input, self.mode)
            input = output
        return output

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update_rmsprop(self, lr, rho=0.999, epsi=10e-05):
        # self.r_w = np.zeros(self.W.shape)    # initialise r for w and b
        # self.r_b = np.zeros(self.b.shape)

        for layer in self.layers:
            layer.r_w = rho * layer.r_w + (1 - rho) * layer.grad_W * layer.grad_W
            layer.W = layer.W - (lr / (np.sqrt(layer.r_w) + epsi) * layer.grad_W)  # update weight

            # weight_decay: float = 0.00005  # 1e-5                                                    # weight decay
            # layer.W = layer.W - layer.W * weight_decay

            layer.r_b = rho * layer.r_b + (1 - rho) * layer.grad_b * layer.grad_b
            layer.b = layer.b - (lr / (np.sqrt(layer.r_b) + epsi) * layer.grad_b)  # update bias

            layer.grad_W = np.zeros_like(layer.grad_W)
            layer.grad_b = np.zeros_like(layer.grad_b)

            # update BN param
            if layer.bn and not layer.isOutputLayer:
                layer.bn_param['gamma'] = layer.bn_param['gamma'] - lr * layer.bn_param['dgamma']
                layer.bn_param['beta'] = layer.bn_param['beta'] - lr * layer.bn_param['dbeta']

    def update_momentum(self, lr, momentum=0.9, weight_decay=0):
        for layer in self.layers:
            layer.V_w = momentum * layer.V_w + lr * layer.grad_W
            layer.V_b = momentum * layer.V_b + lr * layer.grad_b
            layer.W = layer.W - layer.V_w
            weight_decay: float = 1e-5
            layer.W = layer.W - lr * layer.W * weight_decay
            layer.b = layer.b - layer.V_b

            # update BN param
            if layer.bn and not layer.isOutputLayer:
                layer.bn_param['gamma'] = layer.bn_param['gamma'] - lr * layer.bn_param['dgamma']
                layer.bn_param['beta'] = layer.bn_param['beta'] - lr * layer.bn_param['dbeta']

    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            weight_decay: float = 0.02  # 1e-5                                                    # weight decay
            layer.W = layer.W - lr * layer.W * weight_decay
            layer.b -= lr * layer.grad_b
            # update BN param
            if layer.bn and not layer.isOutputLayer:
                layer.bn_param['gamma'] = layer.bn_param['gamma'] - lr * layer.bn_param['dgamma']
                layer.bn_param['beta'] = layer.bn_param['beta'] - lr * layer.bn_param['dbeta']

    def fit(self, X, y, batch_size=16, learning_rate=0.1, epochs=100, train_test_split=0.8, data_val="data_test",
            label_val="label_test", momentum=0.9, weight_decay=0):
        print("**" * 50)
        print("Model layers: {}".format(self.n_neurons))
        print("Model activation functions: {}".format(self.activation))
        print("Batch size: {}".format(batch_size))
        print("Total epoch: {}".format(epochs))
        print("Learning rate: {}".format(learning_rate))
        print("Dropout: {}".format(self.dropout))
        print("Batch Norm: {}".format(self.bn))
        print("Momentum: {}".format(momentum))
        print("Weight_decay: {}".format(weight_decay))
        print("**" * 50)

        # # create a dic
        # # current date and time
        # timestamp = datetime.timestamp(datetime.now())
        # dic_path = './dist/{}/'.format(timestamp)
        # if not os.path.exists(os.path.dirname(dic_path)):
        #     try:
        #         os.makedirs(os.path.dirname(dic_path))
        #     except OSError as exc:  # Guard against race condition
        #         if exc.errno != errno.EEXIST:
        #             raise

        n_data = X.shape[0]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split
        self.n_class = np.unique(y).shape[0]
        loss_epoch = np.zeros(epochs)  # loss of each epoch

        training_acc_arr = []
        val_acc_arr = []
        training_loss_arr = []
        val_loss_arr = []
        epoch_time_arr = []
        for epoch in range(epochs):
            # calculate running time of one epoch
            epoch_start = datetime.now()
            # train mode
            self.train()

            # shuffle the training set to make each epoch's batch different
            shuffle = np.arange(X.shape[0])
            np.random.shuffle(shuffle)
            X_train = X[shuffle]
            y_train = y[shuffle]

            loss_sum = 0  # the loss summation for this epoch
            batch_acc = []
            for batch_start_ind in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[batch_start_ind: min(batch_start_ind + batch_size, X_train.shape[0])]
                y_batch = y_train[batch_start_ind: min(batch_start_ind + batch_size, X_train.shape[0])]
                # forward pass
                y_hat = self.forward(X_batch)

                # backward pass
                loss, delta = nn.CrossEntropy().criterion_CrossEntropy(y_batch, y_hat)
                self.backward(delta)

                # update
                self.update_momentum(learning_rate, momentum=momentum, weight_decay=weight_decay)
                # self.update_rmsprop(learning_rate, rho=0.999, epsi=10e-05)
                # self.update(learning_rate)

                # Epoch loss sum
                loss_sum += loss.sum()

            loss_epoch[epoch] = loss_sum

            self.eval()
            y_hat_train = self.forward(X_train)
            y_hat_val = self.forward(data_val)
            self.training_acc = self.accuracy(y_train, y_hat_train)
            self.val_acc = self.accuracy(label_val, y_hat_val)
            loss_train, _ = nn.CrossEntropy().criterion_CrossEntropy(y_train, y_hat_train)
            loss_val, _ = nn.CrossEntropy().criterion_CrossEntropy(label_val, y_hat_val)

            training_acc_arr.append(self.training_acc)
            val_acc_arr.append(self.val_acc)
            training_loss_arr.append(loss_train)
            val_loss_arr.append(loss_val)
            epoch_time_arr.append(datetime.now() - epoch_start)

            print(
                'Epoch ({}/{}): Train Loss: {}  Train Accuracy: {}  Validation Loss: {}  Validation Accuracy: {}'.format(
                    epoch + 1,
                    epochs,
                    loss_sum,
                    self.training_acc,
                    loss_val,
                    self.val_acc))
        return self

    def accuracy(self, y, y_hat):
        y_pred = np.argmax(y_hat, axis=1).reshape(y.shape)
        accuracy = np.equal(y, y_pred).sum() / y.shape[0]
        return accuracy

    def predict(self, X):
        y_hat = self.forward(X)
        y_pred = np.argmax(y_hat, axis=1)
        return y_pred
