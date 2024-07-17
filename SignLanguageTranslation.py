'''
video frame size: 1080 x 1920
webcam frame size: 480 x 640
'''
import cv2
import os
import mediapipe as mp
import numpy as np
import shutil
from tensorflow.keras.models import Sequential #sequential nn
from tensorflow.keras.layers import LSTM, Dense #for action detection, hidden layers

#mediapipe holistic models
mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utilities

#actual feature detection function
def detect_features(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

#for drawing image
def draw_landmarks(image, result):
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 2, circle_radius = 2),
                              mp_drawing.DrawingSpec(color = (255, 255, 255), thickness = 2, circle_radius = 2))
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 2, circle_radius = 2),
                              mp_drawing.DrawingSpec(color = (255, 255, 255), thickness = 2, circle_radius = 2))
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 2, circle_radius = 2),
                              mp_drawing.DrawingSpec(color = (255, 255, 255), thickness = 2, circle_radius = 2))

#actual feature extraction
def extract_features(result):
    pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4)
    left_hand_landmarks = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    right_hand_landmarks = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])

LANDMARK_PATH = 'extracted_features'
actions = os.listdir(LANDMARK_PATH)

#use previous model
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
sequence = []
prev_statement = 'NONE'
curr_statement = 'NONE'
threshold = 0.9
prediction = []
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read() #read frame
        if(not success): #abort incase not opening
            cap.release()
            raise Exception('Cannot read webcam')
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        frame, result = detect_features(frame, holistic) #detect features
        draw_landmarks(frame, result) #draw result on frame
        features = extract_features(result) #extract features
        sequence.append(features) #cosider latest 25 sequences
        sequence = sequence[-40 :]
        cv2.rectangle(frame, (0,0), (1920, 40), (0, 0, 0), -1)
        if len(sequence) == 40:
            ans = model.predict(np.expand_dims(sequence, axis=0))[0] #resize dimensions of sequence for prediction
            prediction.append(np.argmax(ans))
            prediction = prediction[-10:]
            if ans[np.argmax(ans)] >= threshold and np.unique(prediction)[0] == np.argmax(ans): #check with threshold & last 25 predictions are the same
                curr_statement = actions[np.argmax(ans)] + ' ' + str(ans[np.argmax(ans)])
                cv2.putText(frame, curr_statement, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow('Webcam feed', frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()