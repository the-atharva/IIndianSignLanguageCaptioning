'''
dimensions: total #videos, 40(frames per video), 258(total features)

'''
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from tensorflow.keras.models import Sequential #sequential nn
from tensorflow.keras.layers import LSTM, Dense #for action detection, hidden layers
from tensorflow.keras.callbacks import TensorBoard #monitoring model while training
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score #accuracy
import shutil

LANDMARK_PATH = 'extracted_features'
actions = os.listdir(LANDMARK_PATH)
label_map = {label: num for num, label in enumerate(actions)}

total_frames = 40
sequences, labels = [], []
actions = os.listdir(LANDMARK_PATH)
actions = np.array(actions)
for action in actions:
    videos = os.listdir(os.path.join(LANDMARK_PATH, action))
    for video in videos:
        video_arr = []
        for frame in range(1, total_frames + 1):
            frame_arr = np.load(os.path.join(LANDMARK_PATH, action, video, 'frame{}.npy'.format(frame)))
            video_arr.append(frame_arr)
        sequences.append(video_arr)
        labels.append(label_map[action])
X = np.array(sequences) # total videos, 25, 258
y = to_categorical(labels).astype(int)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

TESTING_PATH = 'testing_arrays'
try:
    shutil.rmtree(TESTING_PATH)
except:
    pass
os.makedirs(TESTING_PATH)
np.save(os.path.join(TESTING_PATH, 'actions'), X_test)
np.save(os.path.join(TESTING_PATH, 'results'), y_test)

log_dir = os.path.join('logs')
tensorboard_callback = TensorBoard(log_dir = log_dir)
model = Sequential()
activation_fn = 'sigmoid'

model.add(LSTM(64, return_sequences=True, activation=activation_fn, input_shape=(X.shape[1], X.shape[2]))) #automate
model.add(LSTM(128, return_sequences=True, activation=activation_fn))
model.add(LSTM(64, return_sequences=False, activation=activation_fn))
model.add(Dense(64, activation=activation_fn))
model.add(Dense(32, activation=activation_fn))
model.add(Dense(actions.shape[0], activation='softmax'))    

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=1200, callbacks=[tensorboard_callback])
print(model.summary())

MODEL = 'model.keras'
try:
    os.remove(MODEL)
except:
    pass

model.save(MODEL)
