'''
Tests accuracy of data on testing dataset.
Generates confusion matrix
'''
import numpy as np
import os
from tensorflow.keras.models import Sequential #sequential nn
from tensorflow.keras.layers import LSTM, Dense #for action detection, hidden layers
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, ConfusionMatrixDisplay #accuracy
import matplotlib.pyplot as plt

LANDMARK_PATH = 'extracted_features'
actions = os.listdir(LANDMARK_PATH)
label_map = {label: num for num, label in enumerate(actions)}

TESTING_PATH = 'testing_arrays'
X_test = np.load(os.path.join(TESTING_PATH, 'actions.npy'))
y_test = np.load(os.path.join(TESTING_PATH, 'results.npy'))

model = Sequential()
activation_fn = 'sigmoid'
model.add(LSTM(64, return_sequences=True, activation=activation_fn, input_shape=(40, 258))) #automate
model.add(LSTM(128, return_sequences=True, activation=activation_fn))
model.add(LSTM(64, return_sequences=False, activation=activation_fn))
model.add(Dense(64, activation=activation_fn))
model.add(Dense(32, activation=activation_fn))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('model.keras')

y_true = np.argmax(y_test, axis=1).tolist()
y_hat = model.predict(X_test)
y_hat = np.argmax(y_hat, axis=1).tolist()

print('Total accuracy: {}%'.format(str(accuracy_score(y_true, y_hat) * 100)))
print('\nConfusion Matrix')
matrix = confusion_matrix(y_true, y_hat)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=actions)
disp.plot()
plt.xticks(rotation=45)
plt.show()
category_matrix = multilabel_confusion_matrix(y_true, y_hat)
print('\nMultilabel confusion matrix: ')
for i in range(10):
    correct = category_matrix[i][1][1]
    total = category_matrix[i][0][1] + correct
    print('{0}:\tpercentage correct = {1}%'.format(actions[i], str(float(correct)*100/total)))
print('\nConfusion matrix')
print(category_matrix)