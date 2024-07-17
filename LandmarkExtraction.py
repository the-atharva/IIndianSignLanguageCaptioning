'''
current frame size: 1080 x 1920
webcam frame size: 480 x 640
Around 360 videos
'''
import cv2
import os
import mediapipe as mp
import numpy as np
import shutil

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
    #applies results to image (pose, left, right) connectors
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
    #flattens everything into single array
    pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4)
    left_hand_landmarks = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    right_hand_landmarks = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])

# DATA_PATH = '../raw data/ISL_CSLRT_Corpus/Videos_Sentence_Level'
DATA_PATH = 'video data'
LANDMARK_PATH = 'extracted_features'
try:
    shutil.rmtree(LANDMARK_PATH)
except:
    pass

with mp_holistic.Holistic(min_tracking_confidence = 0.5, min_detection_confidence = 0.5) as model:
    actions = os.listdir(DATA_PATH) #all actions in folder
    for action in actions:
        videos = os.listdir(os.path.join(DATA_PATH, action)) #all videos for action
        for video in videos:
            try:
                cap = cv2.VideoCapture(os.path.join(DATA_PATH, action, video))
            except:
                print('Unable to read action: ' + action + ' video: ' + video)
            #take video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_time = total_frames/fps
            #set prescale
            frame_num = 40 #frames per video
            if(total_frames < frame_num):
                print("Unable to process video" + video)
                continue
            prescale = total_frames/frame_num
            select_frame = 0
            ctr = 1
            try:
                os.makedirs(os.path.join(LANDMARK_PATH, action, video[:-4])) #make folder for saving landmarks
            except:
                pass
            #actual iteration
            for i in range(total_frames):
                success, frame = cap.read() #read frame
                frame, result = detect_features(frame, model) #detect features
                draw_landmarks(frame, result) #draw result on frame
                landmarks = extract_features(result) #extract landmarks
                if i == int(select_frame):
                    np.save(os.path.join(LANDMARK_PATH, action, video[:-4], 'frame' + str(ctr)), landmarks)
                    ctr += 1
                    select_frame += prescale
                cv2.imshow("Video", frame)
                if cv2.waitKey(10) & 0xff == ord('q'):
                    break
            cap.release() 
















