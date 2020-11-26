import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.metrics import classification_report

x_train_smpl = "x_train_gr_smpl_reduced.csv"
y_train_smpl = "y_train_smpl.csv"

x_data = pd.read_csv(x_train_smpl)
y_data = pd.read_csv(y_train_smpl)

data_array = x_data.values
labels_array = y_data.values 

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(data_array):
    #print("Train:", train_index, "Test:", test_index)
    X_train, X_test = data_array[train_index], data_array[test_index]
    y_train, y_test = labels_array[train_index], labels_array[test_index]


# (trainData, testData, trainLabels, testLabels) = KFold(
#     data_array, labels_array, test_size = 0.15, random_state=42
# ) #75% for training 25% for testing

target_names = ['20','30','50','60','70','left turn', 'right turn',
                'predestrian crossing', 'children', 'cycle route ahead']

#train the linear classification
print("[INFO] training Linear classifier...")
model = LinearSVC()
model.fit(X_train, y_train.ravel())

#evaluate the classifier
print("[INFO] evaluating classifier...")
predications = model.predict(X_test)


print("The first five prediction {}".format(predications))
print("The real first five labels {}".format(y_test.ravel()))

mse = metrics.mean_squared_error(y_test, predications)
print("Mean Squared Error {}".format(mse))


print(classification_report(y_test.ravel(), predications, target_names=target_names))

label_numbers = [0,1,2,3,4,5,6,7,8,9]

print("confusion_matrix")
print(confusion_matrix(y_test, predications, labels = label_numbers))

