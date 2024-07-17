'''
Calculates the 10 shortest actions by total frames by mean or median of all frames
video frame size: 1080 x 1920
webcam frame size: 480 x 640
'''
import cv2
import os
import shutil
import statistics

DATA_PATH = 'video data'


total_actions = 10
actions = os.listdir(DATA_PATH) #all actions in folder
arr = {}
result = {}
for action in actions:
    arr[action] = []
    result[action] = []
    videos = os.listdir(os.path.join(DATA_PATH, action)) #all videos for action
    for video in videos:
        cap = cv2.VideoCapture(os.path.join(DATA_PATH, action, video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        arr.get(action).append(total_frames)
        cap.release() 
    result.get(action).append(statistics.mean(arr.get(action)))
    result.get(action).append(statistics.median(arr.get(action)))

temp = {}
for i in result:
    temp[result.get(i)[1]] = i

keys = list(temp.keys())
keys.sort()
ans = {temp.get(i): i for i in keys}

final_actions = list(ans.keys())[: total_actions]

for i in final_actions:
    print(str(i) + ' ' + str(ans.get(i)))

for action in actions:
    if action not in final_actions:
        try:
            shutil.rmtree(os.path.join(DATA_PATH, action))
        except:
            print('could not delete: ' + action)












