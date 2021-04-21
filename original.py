import numpy as np
from sklearn.utils import shuffle
import torch
import math

train_data = np.load("./Assignment1-Dataset/train_data.npy")
train_label = np.load("./Assignment1-Dataset/train_label.npy")
test_label = np.load("./Assignment1-Dataset/test_label.npy")
test_data = np.load("./Assignment1-Dataset/test_data.npy")

class CrossEntropy(object):
    def criterion_CrossEntropy(self, y, y_hat):
        self.n_class = y_hat.shape[-1]
        y_hat_prob = self.softmax(y_hat)
        y_one_hot = np.array(self.oneHotEncoding(y))
        eps = np.finfo(float).eps  # in case of infinite log
        loss = -np.sum(y_one_hot * np.log(y_hat_prob + eps))
        delta = y_hat_prob - y_one_hot
        return loss, delta


    def oneHotEncoding(self, data):
        # data (batch_size,)
        onehot = np.zeros((data.shape[0], self.n_class))
        for i in range(data.shape[0]):
            k = int(data[i])
            onehot[i, k] = int(1)
        return onehot.squeeze()  # len(y) * 10 features


    def softmax(self, data):
        if data.ndim >= 2:
            data = data - data.max(axis=1).reshape(-1, 1)
        else:
            data = data - data.max()# in case of overflow
        denominator = np.exp(data).sum()
        output = np.exp(data) / denominator
        return output

class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        a = np.tanh(x)
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x)
        return a * (1 - a)

    def __relu(self, x):
        return np.maximum(0, x)  # max(0,x)

    def __relu_deriv(self, a):
        return np.where(a > 0, 1, 0)

    def __init__(self, activation='tanh'):
        if activation == 'logistic':
            self.name = 'logistic'
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.name = 'tanh'
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.name = 'relu'
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv

class Linear(object):
    def __init__(self, n_in, n_out,
                 activation_last_layer='relu', activation='relu', W=None, b=None):
        self.input = None
        if activation:
            self.activation = Activation(activation).f
        self.n_in = n_in  # number of nurons in current layers
        self.n_out = n_out  # number of nurons in next layers

        # initialization
        if activation == 'relu':
            self.kaiming_init()
        else:
            self.xaiver_init()

        self.V_w = np.zeros(self.W.shape)  # momentum for W
        self.V_b = np.zeros(self.b.shape)  # momentum for b

        # activation deriv of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

    def kaiming_init(self):
        # self.W = np.random.normal(
        #     loc=0,
        #     scale=2 / self.n_in,
        #     size=(self.n_in, self.n_out)
        # )

        self.W = (torch.randn(self.n_in, self.n_out) * math.sqrt(2 / self.n_in)).numpy()

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(shape=(self.n_out,))

        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def xaiver_init(self):
        # we randomly assign small values for the weights as the initiallization
        self.W = np.random.uniform(
            low=-np.sqrt(6. / (self.n_in + self.n_out)),
            high=np.sqrt(6. / (self.n_in + self.n_out)),
            size=(self.n_in, self.n_out)
        )
        if self.activation == Activation('logistic').f:
            self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(shape=(self.n_out,))

        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input):
        lin_output = input @ self.W + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        return self.output

    def backward(self, delta):
        batch_size = delta.shape[0]
        output = []
        for i in range(batch_size):
            output.append(np.atleast_2d(self.input[i]).T.dot(np.atleast_2d(delta[i])))
        output = np.stack(output, axis=0)
        self.grad_W = np.mean(output, axis=0)
        self.grad_b = np.mean(delta, axis=0)
        if self.activation_deriv:
            delta = (delta @ self.W.T) * self.activation_deriv(self.input)
            return delta
        else:
            return delta @ self.W.T

class MLP:
    def __init__(self, layers, activation=[None, 'tanh', 'tanh']):
        self.layers = []
        self.params = []

        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(Linear(layers[i], layers[i + 1], activation[i], activation[i + 1]))

    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    def backward(self, delta):
        delta = self.layers[-1].backward(delta)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def update_momentum(self, lr, momentum=0.9):
        for layer in self.layers:
            layer.V_w = momentum * layer.V_w + (1 - momentum) * layer.grad_W
            layer.V_b = momentum * layer.V_b + (1 - momentum) * layer.grad_b
            layer.W = layer.W - lr * layer.V_w
            layer.b = layer.b - lr * layer.V_b
            layer.grad_W = np.zeros_like(layer.grad_W)
            layer.grad_b = np.zeros_like(layer.grad_b)

    def fit(self, X, y, batch_size=16, learning_rate=0.1, epochs=100):
        X = np.array(X)
        y = np.array(y)
        self.n_class = np.unique(y).shape[0]
        loss_epoch = np.zeros(epochs)  # loss of each epoch
        """
        for mini-batch
        """
        for epoch in range(epochs):
            # shuffle the training set to make each epoch's batch different
            X, y = shuffle(X, y)
            loss_sum = 0  # the loss summation for this epoch
            for batch_start_ind in range(0, X.shape[0], batch_size):
                X_batch = X[batch_start_ind: min(batch_start_ind + batch_size, X.shape[0])]
                y_batch = y[batch_start_ind: min(batch_start_ind + batch_size, X.shape[0])]
                # forward pass
                y_hat = self.forward(X_batch)

                # backward pass
                loss, delta = CrossEntropy().criterion_CrossEntropy(y_batch, y_hat)
                self.backward(delta)

                # update
                self.update_momentum(learning_rate, momentum=0.9)

                # Epoch loss sum
                loss_sum += loss.sum()
            loss_epoch[epoch] = loss_sum

            if epoch % 1 == 0:
                print("--" * 50)
                print("Loss of Epoch {}/{}: {}".format(epoch+1, epochs, loss_sum))
                print("Training acc of Epoch {}/{}: {}".format(epoch+1, epochs, self.accuracy(y, self.forward(X))))

        return loss_epoch

    def accuracy(self, y, y_hat):
        y_pred = np.argmax(y_hat, axis=1).reshape(y.shape)
        accuracy = np.equal(y, y_pred).sum() / y.shape[0]
        return accuracy

    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        return output

### Try different MLP models
n_class = np.unique(train_label).shape[0]
nn = MLP([128, 256, 10], [None, None, None])
input_data = train_data[0:1000]
output_data = train_label[0:1000]

cross_entropy = nn.fit(input_data, output_data, learning_rate=0.01, epochs=5000, batch_size = 1)


