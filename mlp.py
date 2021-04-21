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


class MLP:
    def __init__(self, n_neurons=[128, 256, 10], activation=[None, 'relu', 'relu'], dropout=0.2, bn=True):
        self.current_epoch = 0
        self.bn = bn
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.activation = activation
        self.mode = 'train'
        self.layers = []
        for i in range(len(n_neurons) - 1):
            self.layers.append(nn.Linear(n_in=n_neurons[i],
                                         n_out=n_neurons[i + 1],
                                         activation=self.activation[i+1],
                                         activation_last_layer=self.activation[i],
                                         layer_index=i,
                                         bn=self.bn,
                                         dropout=self.dropout,
                                         output_layer=(i==len(n_neurons) - 2)))

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

            weight_decay: float = 0.00005  # 1e-5                                                    # weight decay
            layer.W = layer.W - layer.W * weight_decay

            layer.r_b = rho * layer.r_b + (1 - rho) * layer.grad_b * layer.grad_b
            layer.b = layer.b - (lr / (np.sqrt(layer.r_b) + epsi) * layer.grad_b)  # update bias

            layer.grad_W = np.zeros_like(layer.grad_W)
            layer.grad_b = np.zeros_like(layer.grad_b)

            # update BN param
            if layer.bn and not layer.isOutputLayer:
                layer.bn_param['gamma'] = layer.bn_param['gamma'] - lr * layer.bn_param['dgamma']
                layer.bn_param['beta'] = layer.bn_param['beta'] - lr * layer.bn_param['dbeta']

    def update_momentum(self, lr, momentum=0.9):
        for layer in self.layers:
            # weight_decay: float = 1e-5
            layer.V_w = momentum * layer.V_w + (1 - momentum) * layer.grad_W
            layer.V_b = momentum * layer.V_b + (1 - momentum) * layer.grad_b
            layer.W = layer.W - lr * layer.V_w
            # layer.W = layer.W - layer.W * weight_decay
            layer.b = layer.b - lr * layer.V_b
            layer.grad_W = np.zeros_like(layer.grad_W)
            layer.grad_b = np.zeros_like(layer.grad_b)

            # update BN param
            if layer.bn and not layer.isOutputLayer:
                layer.bn_param['gamma'] = layer.bn_param['gamma'] - lr * layer.bn_param['dgamma']
                layer.bn_param['beta'] = layer.bn_param['beta'] - lr * layer.bn_param['dbeta']

    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
            # update BN param
            if layer.bn and not layer.isOutputLayer:
                layer.bn_param['gamma'] = layer.bn_param['gamma'] - lr * layer.bn_param['dgamma']
                layer.bn_param['beta'] = layer.bn_param['beta'] - lr * layer.bn_param['dbeta']

    def fit(self, X, y, batch_size=16, learning_rate=0.1, epochs=100, train_test_split=0.8, data_val="data_test",
            label_val="label_test"):
        print("**" * 50)
        print("Model layers: {}".format(self.n_neurons))
        print("Model activation functions: {}".format(self.activation))
        print("Batch size: {}".format(batch_size))
        print("Total epoch: {}".format(self.current_epoch + epochs))
        print("Learning rate: {}".format(learning_rate))
        print("Train_test_spit: {}".format(train_test_split))
        print("Dropout: {}".format(self.dropout))
        print("Batch Norm: {}".format(self.bn))
        print("**" * 50)

        # create a dic
        # current date and time
        timestamp = datetime.timestamp(datetime.now())
        dic_path = './dist/{}/'.format(timestamp)
        if not os.path.exists(os.path.dirname(dic_path)):
            try:
                os.makedirs(os.path.dirname(dic_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        n_data = X.shape[0]
        self.batch_size = batch_size
        self.total_epochs = int(self.current_epoch + epochs)
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split
        self.n_class = np.unique(y).shape[0]
        loss_epoch = np.zeros(epochs)  # loss of each epoch

        loss_history_train = []
        loss_history_test = []
        batch_acc = []
        for epoch in range(epochs):
            # calculate running time of one epoch
            epoch_start = datetime.now()
            # train mode
            self.train()
            # X_train = np.array(X[:int(n_data * train_test_split)])
            # y_train = np.array(y[:int(n_data * train_test_split)])
            # X_val = np.array(X[int(n_data * train_test_split):])
            # y_val = np.array(y[int(n_data * train_test_split):])
            # shuffle the training set to make each epoch's batch different
            X_train = X
            y_train = y
            shuffle = np.arange(X_train.shape[0])
            np.random.shuffle(shuffle)
            X_train = X_train[shuffle]
            y_train = y_train[shuffle]

            loss_sum = 0  # the loss summation for this epoch
            for batch_start_ind in range(0, X_train.shape[0], batch_size):
                current_batch_acc = []
                X_batch = X_train[batch_start_ind: min(batch_start_ind + batch_size, X_train.shape[0])]
                y_batch = y_train[batch_start_ind: min(batch_start_ind + batch_size, X_train.shape[0])]
                # forward pass
                y_hat = self.forward(X_batch)

                # backward pass
                loss, delta = nn.CrossEntropy().criterion_CrossEntropy(y_batch, y_hat)
                self.backward(delta)

                # update
                # self.update_momentum(learning_rate, momentum=0.9)
                self.update_rmsprop(learning_rate, rho=0.999, epsi=10e-05)
                # self.update(learning_rate)

                # Epoch loss sum
                loss_sum += loss.sum()

                # print val acc for every update
                batch_acc_val = self.accuracy(label_val, self.forward(data_val))
                batch_acc.append(batch_acc_val)
                # print("Validation acc of Batch {}/{}: {}".format(math.ceil(batch_start_ind / batch_size), int(X_train.shape[0]/batch_size), self.accuracy(label_val, self.forward(data_val))))

                if batch_start_ind >= 100 * batch_size:
                    n_batch = batch_start_ind / 16  # math.ceil(batch_start_ind / batch_size)
                    print(batch_acc)
                    plt.plot(batch_acc)  # batch - vali - acc
                    plt.xlabel('Batch')
                    plt.ylabel('Accuracy')
                    plt.xlim(0, n_batch)
                    plt.title("MLP with Batch Normalisation")
                    plt.show()
                    time.sleep()

            loss_epoch[epoch] = loss_sum

            self.eval()
            self.training_acc = self.accuracy(y_train, self.forward(X_train))
            # self.val_acc = self.accuracy(label_val, self.forward(data_val))
            y_hat_val = self.forward(data_val)
            loss_val, _ = nn.CrossEntropy().criterion_CrossEntropy(label_val, y_hat_val)
            self.val_acc = self.accuracy(label_val, y_hat_val)
            self.current_epoch += 1
            print(
                'Epoch ({}/{}): Train Loss: {}  Train Accuracy: {}  Validation Loss: {}  Validation Accuracy: {}'.format(
                    self.current_epoch,
                    self.total_epochs,
                    loss_sum,
                    self.training_acc,
                    loss_val,
                    self.val_acc))

            # for train/test loss plot
            loss_history_train.append(loss_sum)
            loss_history_test.append(loss_val)
            loss_history = {'train': loss_history_train, 'test': loss_history_test}
            # print('loss_history.dic',loss_history)

            # save the model to disk
            with open(os.path.join(dic_path,
                                   '#dropout={}#batch_size={}#lr={}#bn={}#epoch_{}#val_acc={}.pkl'.format(self.dropout,
                                                                                                      self.batch_size,
                                                                                                      self.learning_rate,
                                                                                                      self.bn,
                                                                                                      epoch + 1,
                                                                                                      self.val_acc)),
                      "wb") as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)



        # # plot
        # plt.plot(loss_history['train'])
        # plt.plot(loss_history['test'])
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.xlim(0, epoch)
        # plt.show()

        return self

    def accuracy(self, y, y_hat):
        y_pred = np.argmax(y_hat, axis=1).reshape(y.shape)
        accuracy = np.equal(y, y_pred).sum() / y.shape[0]
        return accuracy

    def predict(self, X):
        y_hat = self.forward(X)
        y_pred = np.argmax(y_hat, axis=1)
        return y_pred
