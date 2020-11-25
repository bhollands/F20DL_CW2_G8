from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

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


x_train_smpl = "x_train_gr_smpl_reduced.csv"
y_train_smpl = "y_train_smpl.csv"

x_data = pd.read_csv(x_train_smpl)
y_data = pd.read_csv(y_train_smpl)

data_array = x_data.values
labels_array = y_data.values 

(trainData, testData, trainLabels, testLabels) = train_test_split(
    data_array, labels_array, test_size = 0.15, random_state=42
) #75% for training 25% for testing



def extract_color_hostogram(image, bins=(8,8,8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)
    
    return hist.flatten()


target_names = ['20','30','50','60','70','left turn', 'right turn',
                'predestrian crossing', 'children', 'cycle route ahead']


#train the linear classification
print("[INFO] training Linear classifier...")
model = LogisticRegression()
model.fit(trainData, trainLabels.ravel())

#evaluate the classifier
print("[INFO] evaluating classifier...")
predications = model.predict(testData)


print("The first five prediction {}".format(predications))
print("The real first five labels {}".format(testLabels.ravel()))

mse = metrics.mean_squared_error(testLabels, predications)
print("Mean Squared Error {}".format(mse))


print(classification_report(testLabels.ravel(), predications, target_names=target_names))

label_numbers = [0,1,2,3,4,5,6,7,8,9]

print("confusion_matrix")
print(confusion_matrix(testLabels, predications, labels = label_numbers))

scores = cross_val_score(model, data_array, labels_array.ravel(), cv = 10)
print("Accuracy: %0.f (+/- %0.2f)" % (scores.mean(), scores.std()*2))