import numpy as np
import mlp
import pickle
np.random.seed(5329)

train_data = np.load("./Assignment1-Dataset/train_data.npy")
train_label = np.load("./Assignment1-Dataset/train_label.npy")
test_label = np.load("./Assignment1-Dataset/test_label.npy")
test_data = np.load("./Assignment1-Dataset/test_data.npy")

n_class = np.unique(train_label).shape[0]
n_features = train_data.shape[1]
nn = mlp.MLP([n_features, 128, 128, n_class],
             [None, 'relu', 'relu',  'relu'],
             dropout=0,
             bn=False)

input_data = train_data
input_data -= np.mean(input_data, axis=0)  # Mean subtraction
input_data /= np.std(input_data, axis=0)  # Scaling
output_data = train_label

model = nn.fit(input_data,
               output_data,
               learning_rate=0.001,
               epochs=30,
               batch_size=1,
               train_test_spit=0.8)

model.eval()
accuracy = model.accuracy(test_label, model.forward(test_data))
print("**" * 50)
print("Training acc of Test data: {}".format(accuracy))

# with open('./models/epoch_1.pkl', 'rb') as input:
#     model = pickle.load(input)
#     print(model.training_acc)
