import numpy as np
import matplotlib.pyplot as plt

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


def confusion_matrix(y, y_hat, th):

    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(len(y_hat)):
        if y_hat[i] < th:
            if y[i] != 0:
                fn = fn + 1
            else:
                tn = tn + 1
        elif y_hat[i] >= th:
            if y[i] != 1:
                fp = fp + 1
            else:
                tp = tp + 1

    return fp, tp, fn, tn


def f1_score(y, y_hat, th=0.5):
    fp, tp, fn, tn = confusion_matrix(y, y_hat, th)
    f1 = (2 * (tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fn)) + (tp / (tp + fp)))
    return f1


def true_postive_rate(y, y_hat, th):
    fp, tp, fn, tn = confusion_matrix(y, y_hat, th)
    tpr = tp / (tp + fn)
    return tpr


def false_positive_rate(y, y_hat, th):
    fp, tp, fn, tn = confusion_matrix(y, y_hat, th)
    fpr = fp / (tn + fp)
    return fpr


def AUC(y, y_hat):
    thresh_ran = np.arange(0, 1, 0.01)
    pts = []
    for i in thresh_ran:
        fpr = false_positive_rate(y, y_hat, i)
        tpr = true_postive_rate(y, y_hat, i)
        pts.append((fpr, tpr))

    fpr_ary = []
    tpr_ary = []

    for j in range(len(pts) - 1):
        x = pts[j]
        y = pts[j + 1]
        fpr_ary.append([x[1], y[1]])
        tpr_ary.append([x[0], y[0]])

    auc_score = np.sum(np.trapz(tpr_ary, fpr_ary)) + 1
    plt.plot(tpr_ary, fpr_ary, 'c')
    # plt.plot([0, 1], [0, 1])
    plt.title('AUC = %.2f' % auc_score)
    # plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()