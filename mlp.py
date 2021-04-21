import numpy as np
import nn
import pickle
import os
import errno
from datetime import datetime
import math


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
            if self.bn:
                layer.bn_param['gamma'] = layer.bn_param['gamma'] - lr * layer.bn_param['dgamma']
                layer.bn_param['beta'] = layer.bn_param['beta'] - lr * layer.bn_param['beta']

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

    def fit(self, X, y, batch_size=16, learning_rate=0.1, epochs=100, train_test_spit=0.8):
        print("**" * 50)
        print("Model layers: {}".format(self.n_neurons))
        print("Model activation functions: {}".format(self.activation))
        print("Batch size: {}".format(batch_size))
        print("Total epoch: {}".format(epochs))
        print("Learning rate: {}".format(learning_rate))
        print("Train_test_spit: {}".format(train_test_spit))
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
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_test_spit = train_test_spit
        self.n_class = np.unique(y).shape[0]
        loss_epoch = np.zeros(epochs)  # loss of each epoch
        for epoch in range(epochs):
            # calculate running time of one epoch
            epoch_start = datetime.now()
            # train mode
            self.train()

            X_train = np.array(X[:int(n_data * train_test_spit)])
            y_train = np.array(y[:int(n_data * train_test_spit)])
            X_val = np.array(X[int(n_data * train_test_spit):])
            y_val = np.array(y[int(n_data * train_test_spit):])
            # shuffle the training set to make each epoch's batch different
            # shuffle = np.arange(X_train.shape[0])
            # np.random.shuffle(shuffle)
            # X_train = X_train[shuffle]
            # y_train = y_train[shuffle]

            loss_sum = 0  # the loss summation for this epoch
            for batch_start_ind in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[batch_start_ind: min(batch_start_ind + batch_size, X_train.shape[0])]
                y_batch = y_train[batch_start_ind: min(batch_start_ind + batch_size, X_train.shape[0])]
                # forward pass
                y_hat = self.forward(X_batch)

                # backward pass
                loss, delta = nn.CrossEntropy().criterion_CrossEntropy(y_batch, y_hat)
                self.backward(delta)

                # update
                # self.update_momentum(learning_rate, momentum=0.9)
                # self.update(learning_rate)
                self.update_rmsprop(learning_rate)

                # Epoch loss sum
                loss_sum += loss.sum()

                # print val acc for every update
                # print("##" * 50)
                # print("Loss of Batch {}/{}: {}".format(math.ceil(batch_start_ind / batch_size), int(X_train.shape[0]/batch_size), loss.sum()))
                # print("Validation acc of Batch {}/{}: {}".format(math.ceil(batch_start_ind / batch_size), int(X_train.shape[0]/batch_size), self.accuracy(y_val, self.forward(X_val))))
            loss_epoch[epoch] = loss_sum

            self.eval()
            self.training_acc = self.accuracy(y_train, self.forward(X_train))
            self.val_acc = self.accuracy(y_val, self.forward(X_val))
            print("--" * 50)
            print("Loss of Epoch {}/{}: {}".format(epoch + 1, epochs, loss_sum))
            print("Running time of Epoch {}/{}: {}".format(epoch + 1, epochs, datetime.now() - epoch_start))
            print("Training acc of Epoch {}/{}: {}".format(epoch + 1, epochs, self.training_acc))
            print("Validation acc of Epoch {}/{}: {}".format(epoch + 1, epochs, self.val_acc))



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
        return self

    def accuracy(self, y, y_hat):
        y_pred = np.argmax(y_hat, axis=1).reshape(y.shape)
        accuracy = np.equal(y, y_pred).sum() / y.shape[0]
        return accuracy

    def predict(self, X):
        y_hat = self.forward(X)
        y_pred = np.argmax(y_hat, axis=1)
        return y_pred
