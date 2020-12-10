import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

np.random.seed(7)

#import the data
x_train_smpl = "Original datasets/x_train_gr_smpl_reduced.csv"
y_train_smpl = "Original datasets/y_train_smpl.csv"

x_data = pd.read_csv(x_train_smpl)
y_data = pd.read_csv(y_train_smpl)

data_array = x_data.values #X_train
labels_array = y_data.values #y_train

train_indices = np.arange(data_array.shape[0])
np.random.shuffle(train_indices)
data_array = data_array[train_indices]
labels_array = labels_array[train_indices]

kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(data_array):
    #print("Train:", train_index, "Test:", test_index)
    X_train, X_test = data_array[train_index], data_array[test_index]
    y_train, y_test = labels_array[train_index], labels_array[test_index]

class_names = np.array(['20 kph','30 kph','50 kph','60 kph','70 kph','left turn', 'right turn',
                'predestrian crossing', 'children', 'cycle route ahead'])

# #image = X_train[1].reshape(35,35)

#normalising
X_train = X_train / 255.0
X_test = X_test /255.0

y_train.flatten()


EPOCHS = 10

model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(1225)),
    tf.keras.layers.Dense(1225, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

opt = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.03, nesterov=False, name="SGD")

model.compile(optimizer=opt,
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x = X_train, y = y_train, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('\nTest accuracy:', test_acc)

y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool, target_names=class_names))

label_numbers = [0,1,2,3,4,5,6,7,8,9]

print("confusion_matrix")
print(confusion_matrix(y_test, y_pred_bool, labels = label_numbers))

print(class_names[np.argmax(y_pred[3])])
plt.imshow(X_test[3].reshape(35,35))
plt.colorbar()
plt.grid(False)
plt.show()


# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array, true_label.item(i)
#     plt.grid(False)
#     plt.xticks(range(10))
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0,1])
#     predicted_label = np.argmax(predictions_array)
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')

# def plot_image(i, predictions_array, true_label, img):
    
#     predictions_array, true_label, img = predictions_array, true_label[i], img[i].reshape(35,35)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img, cmap=plt.cm.binary)
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'

#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                 100*np.max(predictions_array),
#                                 class_names[true_label]),
#                                 color=color)
                            
# # i = 12
# # plt.figure(figsize=(6,3))
# # plt.subplot(1,2,1)
# # plot_image(i, classifications[i], y_test, X_test)
# # plt.subplot(1,2,2)
# # plot_value_array(i,classifications[i], y_test)
# # plt.show()

# num_rows = 15
# num_cols = 3
# num_images = num_cols*num_rows
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, classifications[i], y_test, X_test)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, classifications[i], y_test)
# plt.tight_layout()
# plt.show()
