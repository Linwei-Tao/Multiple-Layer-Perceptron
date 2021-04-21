import numpy as np
import util


class Linear(object):
    def __init__(self, n_in, n_out, layer_index, activation=None, activation_last_layer = None, bn=True, dropout=0.2,
                 output_layer=False):
        self.isOutputLayer = output_layer
        self.name = "Hidden layer " + str(layer_index + 1) if not self.isOutputLayer else "Output layer"
        self.input = None
        self.n_in = n_in  # number of neurons in current layer
        self.n_out = n_out  # number of neurons in next layer
        self.activation_name = activation
        self.activation_last_layer = activation_last_layer,
        self.bn = bn
        self.dropout = dropout

        # initialize parameters
        # kaiming init (for relu)
        self.W = np.random.normal(
            loc=0,
            scale=2 / self.n_in,
            size=(self.n_in, self.n_out)
        )

        # # random initiallization
        # self.W = np.random.uniform(
        #     low=-np.sqrt(6. / (n_in + n_out)),
        #     high=np.sqrt(6. / (n_in + n_out)),
        #     size=(n_in, n_out)
        # )

        self.b = np.zeros(shape=(self.n_out,))

        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        # momentum parameters
        self.V_w = np.zeros(self.W.shape)
        self.V_b = np.zeros(self.b.shape)

        # rmsprop parameters
        self.r_w = np.zeros(self.W.shape)    # initialise r for w and b
        self.r_b = np.zeros(self.b.shape)

        # batch normalization init
        self.bn_cache = {}
        self.bn_eval = {"running_mean": np.zeros(n_out),
                        "running_var": np.zeros(n_out)}
        self.bn_param = {
            "eps": 1e-5,
            "momentum": 0.9,
            "gamma": np.ones(self.n_out, ),
            "beta": np.zeros(self.n_out, )
        }

    def forward(self, input, mode):
        # print("==" * 50)
        # print("{} forward propagation".format(self.name))
        # print("this layer has {} neurons".format(self.n_out))

        self.mode = mode
        self.input_batch_size = input.shape[0]
        self.input = input
        self.lin_output = self.input @ self.W + self.b
        if self.isOutputLayer:
            self.activation_output = self.activation_forward(self.lin_output)
            return self.activation_output
        self.bn_output = self.bn_forward(self.lin_output)
        self.dropout_output = self.dropout_forward(self.bn_output)
        self.activation_output = self.activation_forward(self.dropout_output)
        return self.activation_output


    def backward(self, delta):
        # print("=="*50)
        # print("{} back propagation".format(self.name))
        # print("this layer has {} neurons".format(self.n_out))

        if not self.isOutputLayer:
            delta = self.activation_backward(delta) # d_dropout
            delta = self.dropout_backward(delta) # d_bn
            delta = self.bn_backward(delta)

        # calculate grad
        grad_W_all = []
        for i in range(self.input_batch_size):
            grad_W_all.append(np.atleast_2d(self.input[i]).T.dot(np.atleast_2d(delta[i])))
        grad_W_all = np.stack(grad_W_all, axis=0)

        self.grad_W = np.mean(grad_W_all, axis=0)
        self.grad_b = np.mean(delta, axis=0)

        delta = delta.dot(self.W.T)  # d_activation
        return delta


    """
    batch normalization functions
    """

    # forward functions
    def bn_forward(self, input):
        if self.bn:
            eps = self.bn_param["eps"]
            momentum = self.bn_param['momentum']
            if self.mode == 'train':
                batch_mean = np.mean(input, axis=0)
                batch_var = np.var(input, axis=0)

                # Estimate running average of mean and variance to use at test time
                self.bn_eval['running_mean'] = momentum * self.bn_eval['running_mean'] + (1 - momentum) * batch_mean
                self.bn_eval['running_var'] = momentum * self.bn_eval['running_var'] + (1 - momentum) * batch_var

                # Normalization followed by Affine transformation
                input_norm = (input - batch_mean) / np.sqrt(batch_var + eps)
                output = self.bn_param["gamma"] * input_norm + self.bn_param["beta"]

                # Cache variables needed during backpropagation
                self.bn_cache["input"] = input
                self.bn_cache["input_norm"] = input_norm
                self.bn_cache["batch_mean"] = batch_mean
                self.bn_cache["batch_var"] = batch_var
                self.bn_cache["gamma"] = self.bn_param["gamma"]
                self.bn_cache["beta"] = self.bn_param["beta"]
                self.bn_cache["eps"] = self.bn_param['eps']

            elif self.mode == 'eval':
                # normalize using running average
                input_norm = (input - self.bn_eval["running_mean"]) / np.sqrt(self.bn_eval["running_var"] + eps)
                output = self.bn_param["gamma"] * input_norm + self.bn_param["beta"]
            return output

        else:
            return input

    # backward functions
    def bn_backward(self, delta):
        if self.bn:
            # unpack
            input_norm = self.bn_cache["input_norm"]
            batch_var = self.bn_cache["batch_var"]
            eps = self.bn_cache["eps"]
            input = self.bn_cache["input"]
            batch_mean = self.bn_cache["batch_mean"]
            gamma = self.bn_cache["gamma"]

            # See derivations above for dgamma, dbeta and dx
            self.bn_param["dgamma"] = np.sum(delta * input_norm, axis=0)
            self.bn_param["dbeta"] = np.sum(delta, axis=0)

            m = input.shape[0]
            t = 1. / np.sqrt(batch_var + eps)

            delta = (gamma * t / m) * (m * delta - np.sum(delta, axis=0)
                                       - t ** 2 * (input - batch_mean)
                                       * np.sum(delta * (input - batch_mean), axis=0))

        return delta

    """
    activation layer functions
    """

    def activation_forward(self, input):
        if self.activation_name == 'tanh':
            return input
        elif self.activation_name == 'relu':
            return np.maximum(0, input)
        elif self.activation_name == 'leaky_relu':
            return np.where(input >= 0, input, 0.01*input)
        else:
            return input

    def activation_backward(self, delta):
        if self.activation_name == 'tanh':
            return delta
        elif self.activation_name == 'relu':
            return delta * np.where(self.dropout_output >= 0, 1, 0)
        elif self.activation_name == 'leaky_relu':
            return delta * np.where(self.dropout_output >= 0, 1, 0.01)
        else:
            return delta

    """
    dropout layer functions
    """

    def dropout_forward(self, input):
        if self.dropout:
            keep_rate = 1 - self.dropout
            self.mask = np.random.binomial(1, keep_rate, size=input.shape) / keep_rate
            output = input * self.mask
            return output
        else:
            return input

    def dropout_backward(self, delta):
        if self.dropout:
            return delta * self.mask
        else:
            return delta


class CrossEntropy(object):
    def __init__(self):
        self.name = "CrossEntropy Loss layer"

    def criterion_CrossEntropy(self, y, y_hat):
        n_class = y_hat.shape[-1]
        y_hat_prob = util.softmax(y_hat)
        y_one_hot = np.array(util.oneHotEncoding(y, n_class))
        eps = np.finfo(float).eps  # in case of infinite log
        loss = -np.sum(y_one_hot * np.log(y_hat_prob + eps))
        delta = y_hat_prob - y_one_hot
        return loss, delta
