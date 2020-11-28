import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

np.random.seed(7)

#import the data
x_train_smpl = "Original datasets/x_train_gr_smpl_reduced.csv"
x_test_smpl = "Original datasets/x_test_gr_smpl_reduced.csv"
y_train_smpl = "Original datasets/y_train_smpl.csv"
y_test_smpl = "Original datasets/y_test_smpl.csv"

x_train_data = pd.read_csv(x_train_smpl)
y_train_data = pd.read_csv(y_train_smpl)
x_test_data = pd.read_csv(x_test_smpl)
y_test_data = pd.read_csv(y_test_smpl)

X_train = x_train_data.values
y_train = y_train_data.values

X_test = x_test_data.values
y_test = y_test_data.values

train_indices = np.arange(X_train.shape[0])
np.random.shuffle(train_indices)
X_train = X_train[train_indices]
y_train = y_train[train_indices]

test_indices = np.arange(X_test.shape[0])
np.random.shuffle(test_indices)
X_test = X_test[test_indices]
y_test = y_test[test_indices]


X_test_df = pd.DataFrame(X_test)
X_test_df.to_csv(r'X_test_random_reduced.csv', index=False)

y_test_df = pd.DataFrame(y_test)
y_test_df.to_csv(r'y_test_random_reduced.csv', index=False)

X_train_df = pd.DataFrame(X_train)
X_train_df.to_csv(r'X_train_random_reduced.csv', index=False)

y_train_df = pd.DataFrame(y_train )
y_train_df.to_csv(r'y_train_random_reduced.csv', index=False)
#np.savetxt("x_train_random_reduced.csv", X_train.astype('int32'), delimiter=",")
