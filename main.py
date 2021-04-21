import numpy as np
import mlp
import pickle
import util


np.random.seed(5329)

train_data = np.load("./Assignment1-Dataset/train_data.npy")
train_label = np.load("./Assignment1-Dataset/train_label.npy")
test_label = np.load("./Assignment1-Dataset/test_label.npy")
test_data = np.load("./Assignment1-Dataset/test_data.npy")

n_class = np.unique(train_label).shape[0]
n_features = train_data.shape[1]
nn = mlp.MLP([n_features, 128, 128, n_class],
             [None, 'relu', 'relu',  None],
             dropout=0,
             bn=True)
# Standardisation normalizer
# standard_transformer = util.Standardisation()
# data_train = standard_transformer.fit_transform(train_data)
# data_test = standard_transformer.transform(test_data)

# min_max normalizer
# min_max_transformer = util.MinMax_transformer()
# data_train = min_max_scaler.fit_transform(train_data)
# data_test = min_max_scaler.transform(test_data)

# without normalizer
data_train = train_data
data_test = test_data
label_train = train_label
label_test = test_label

model = nn.fit(data_train,
               label_train,
               learning_rate=0.001,
               epochs=2,
               batch_size=16,
               train_test_split=0.8,
               data_val=data_test,
               label_val=label_test)

model.eval()
accuracy = model.accuracy(test_label, model.forward(test_data))
print("**" * 50)
print("Training acc of Test data: {}".format(accuracy))

# with open('/Users/dylantao/Documents/PycharmProject/MLP/dist/1619015014.591005/#dropout=0#batch_size=16#lr=0.001#bn=False#epoch_4#val_acc=0.5394.pkl', 'rb') as input:
#     model = pickle.load(input)
#     model = nn.fit(data_train,
#                    label_train,
#                    learning_rate=0.001,
#                    epochs=30,
#                    batch_size=16,
#                    train_test_split=0.8)
