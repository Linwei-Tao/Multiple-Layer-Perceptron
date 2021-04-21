import numpy as np


class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x)
        return a * (1 - a)

    def __relu(self, x):
        return np.maximum(0, x)  # max(0,x)

    def __relu_deriv(self, a):
        return np.array([1 if i > 0 else 0 for i in a])

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


# now we define the hidden layer for the mlp
# for example, h1 = HiddenLayer(10, 5, activation="tanh") means we create a layer with 10 dimension input and 5 dimension output, and using tanh activation function.
# notes: make sure the input size of hiddle layer should be matched with the output size of the previous layer!

class HiddenLayer(object):
    def __init__(self, n_in, n_out,
                 activation_last_layer='tanh', activation='tanh', W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = None
        self.activation = Activation(activation).f

        # activation deriv of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the initiallization
        self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )
        if activation == 'logistic':
            self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(n_out, )

        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    # the forward and backward progress for each training epoch
    # please learn the week2 lec contents carefully to understand these codes.
    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        return self.output

    def backward(self, delta, output_layer=False):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta


class MLP:
    """
    """

    # for initiallization, the code will create all layers automatically based on the provided parameters.
    def __init__(self, layers, activation=[None, 'tanh', 'tanh']):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        ### initialize layers
        self.layers = []
        self.params = []

        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation[i], activation[i + 1]))

    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation[-1]).f_deriv
        # MSE
        error = y - y_hat
        loss = error ** 2
        # calculate the delta of the output layer
        delta = -error * activation_deriv(y_hat)
        # return loss and delta
        return loss, delta

    # backward progress
    def backward(self, delta):
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!
    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b

    # define the training function
    # it will return all losses within the whole training process.
    def fit(self, X, y, learning_rate=0.1, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X = np.array(X)
        y = np.array(y)
        n_data = X.shape[0]
        to_return = np.zeros(epochs)

        split = int(n_data*0.8)
        X_val = X[split:]
        y_val = y[split:]
        X = X[:split]
        y = y[:split]
        for k in range(epochs):
            loss = np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])

                # forward pass
                y_hat = self.forward(X[i])

                # backward pass
                loss[i], delta = CrossEntropy().criterion_CrossEntropy(y[i], y_hat)
                self.backward(delta)

                # update
                self.update(learning_rate)
            to_return[k] = np.mean(loss)
            print("--" * 50)
            self.training_acc = self.accuracy(y, self.forward(X))
            self.val_acc = self.accuracy(y_val, self.forward(X_val))
            print("Training acc of Epoch {}/{}: {}".format(k + 1, epochs, self.training_acc))
            print("Validation acc of Epoch {}/{}: {}".format(k + 1, epochs, self.val_acc))
        return to_return

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        return output

    def accuracy(self, y, y_hat):
        y_pred = np.argmax(y_hat, axis=1).reshape(y.shape)
        accuracy = np.equal(y, y_pred).sum() / y.shape[0]
        return accuracy


class CrossEntropy(object):
    def criterion_CrossEntropy(self, y, y_hat):
        self.n_class = y_hat.shape[-1]
        y_hat_prob = self.softmax(y_hat).squeeze()
        y_one_hot = np.array(self.oneHotEncoding(y, 10))
        eps = np.finfo(float).eps  # in case of infinite log
        loss = -np.sum(y_one_hot * np.log(y_hat_prob + eps))
        delta = y_hat_prob - y_one_hot
        return loss, delta

    def oneHotEncoding(self, data, n_class):
        onehot = np.zeros((data.shape[0], n_class))
        for i in range(data.shape[0]):
            k = int(data[i])
            onehot[i, k] = int(1)
        return onehot.squeeze()  # len(y) * 10 features

    def softmax(self, data):
        # output = []
        # for i in range(0, data.shape[0]):
            # equ = np.exp(data[i]) / np.exp(data).sum()
            # output.append(equ)
        # return np.vstack(output)
        return np.exp(data) / np.exp(data).sum()


import numpy as np
np.random.seed(5329)

train_data = np.load("./Assignment1-Dataset/train_data.npy")
train_label = np.load("./Assignment1-Dataset/train_label.npy")
test_label = np.load("./Assignment1-Dataset/test_label.npy")
test_data = np.load("./Assignment1-Dataset/test_data.npy")

n_class = np.unique(train_label).shape[0]
n_features = train_data.shape[1]
nn = MLP([n_features, 128, 128, n_class],
             [None, 'relu', 'relu', 'relu'])

input_data = train_data
# input_data -= np.mean(input_data, axis=0)  # Mean subtraction
# input_data /= np.std(input_data, axis=0)  # Scaling
output_data = train_label

model = nn.fit(input_data,
               output_data,
               learning_rate=0.001,
               epochs=100)


# model.eval()
# accuracy = model.accuracy(test_label, model.forward(test_data))
# print("**" * 50)
# print("Training acc of Test data: {}".format(accuracy))

# with open('./models/epoch_1.pkl', 'rb') as input:
#     model = pickle.load(input)
#     print(model.training_acc)
#

