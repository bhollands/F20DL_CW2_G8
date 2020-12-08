from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics

import numpy as np
import pandas as pd
import argparse
import imutils
import cv2
import os 


x_train_smpl = "Step 5 Data/X_train_random_reduced.csv"
x_test_smpl = "Step 5 Data/X_test_random_reduced.csv"
y_train_smpl = "Step 5 Data/y_train_random_reduced.csv"
y_test_smpl = "Step 5 Data/y_test_random_reduced.csv"

x_train_data = pd.read_csv(x_train_smpl)
y_train_data = pd.read_csv(y_train_smpl)
x_test_data = pd.read_csv(x_test_smpl)
y_test_data = pd.read_csv(y_test_smpl)

X_train = x_train_data.values
y_train = y_train_data.values

X_test = x_test_data.values
y_test = y_test_data.values

'''
Randomising
'''
train_indices = np.arange(X_train.shape[0])
np.random.shuffle(train_indices)
X_train = X_train[train_indices]
y_train = y_train[train_indices]

test_indices = np.arange(X_test.shape[0])
np.random.shuffle(test_indices)
X_test = X_test[test_indices]
y_test = y_test[test_indices]



target_names = ['20','30','50','60','70','left turn', 'right turn',
                'predestrian crossing', 'children', 'cycle route ahead']


#train the linear classification
print("[INFO] training Linear classifier...")
model = LogisticRegression(solver="lbfgs", C=10**10,max_iter=50 , random_state=42)
model.fit(X_train, y_train.ravel())

#evaluate the classifier
print("[INFO] evaluating classifier...")
predications = model.predict(X_test)

# print("The first five prediction {}".format(predications))
# print("The real first five labels {}".format(y_test.ravel()))

# mse = metrics.mean_squared_error(y_test, predications)
# print("Mean Squared Error {}".format(mse))

print("Model Accuracy: ",model.score(X_test, y_test, sample_weight=None))

print(classification_report(y_test.ravel(), predications, target_names=target_names))

label_numbers = [0,1,2,3,4,5,6,7,8,9]

print("confusion_matrix")
print(confusion_matrix(y_test, predications, labels = label_numbers))
