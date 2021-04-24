import numpy as np
import mlp

np.random.seed(5329)

train_data = np.load("./Assignment1-Dataset/train_data.npy")
train_label = np.load("./Assignment1-Dataset/train_label.npy")
test_label = np.load("./Assignment1-Dataset/test_label.npy")
test_data = np.load("./Assignment1-Dataset/test_data.npy")

n_class = np.unique(train_label).shape[0]
n_features = train_data.shape[1]
nn = mlp.MLP([n_features, 256, n_class],
             [None, 'relu', 'relu', None],
             dropout=0,
             bn=False)
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
               learning_rate=0.01,
               epochs=3,
               batch_size=64,
               data_val=data_test,
               label_val=label_test,
               momentum=0.9,
               weight_decay=0)



# modules for evaluation
from sklearn import metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
model.eval()
y_pred = model.predict(data_test)
fpr, tpr, thresholds = metrics.roc_curve(label_test, y_pred, pos_label=2)
auc = metrics.auc(fpr, tpr)
f1_score = metrics.f1_score(label_test, y_pred, average='macro')
acc = metrics.accuracy_score(label_test, y_pred)

print("Model AUC: {}".format(auc))
print("Model F1 socre: {}".format(f1_score))
print("Model Accuracy socre: {}".format(acc))

cm = metrics.confusion_matrix(label_test, y_pred)
df_cm = pd.DataFrame(cm, index=[i for i in range(10)],
                     columns=[i for i in range(10)])
plt.figure(figsize=(10, 10))
sn.heatmap(df_cm, annot=True)
plt.show()
