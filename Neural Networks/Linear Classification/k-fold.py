import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

x_train_smpl = "x_train_gr_smpl_reduced.csv"
y_train_smpl = "y_train_smpl.csv"

x_data = pd.read_csv(x_train_smpl)
y_data = pd.read_csv(y_train_smpl)

data_array = x_data.values
labels_array = y_data.values 

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(x_train_smpl):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = x_train_smpl[train_index], x_train_smpl[test_index]
    y_train, y_test = y_train_smpl[train_index], y_train_smpl[test_index]

# (trainData, testData, trainLabels, testLabels) = KFold(
#     data_array, labels_array, test_size = 0.15, random_state=42
# ) #75% for training 25% for testing

