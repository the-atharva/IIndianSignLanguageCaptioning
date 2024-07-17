# IndianSignLanguageCaptioning

Captions 10 phrases from Indian Sign Language in real time from webcam.\

## Abstract
This work is able to transcribe or detect sign language phrases from real time webcam. It is trained on around 300 videos collected from PICT sign language dataset.\
The project uses MediaPipe holistic to extract 258 features from both the hand and torso.\
This process is done for 40 frames per video\
All of these are concatenated into numpy arrays and used to train the LSTM model.\
The LSTM model has 6 layers. 3 LSTM layers and 3 dense layers.\
Then it is tested on around 50 videos.\
Then OpenCv is used to interact with real time webcam to make real time prediction on actions. 

## Structure of the project
The project is divided into different files and each of them have separate purpose.\
It is trained on the dataset is categorised into different folders & all videos for a phrases are in folder with name as that phrase. All these phrases are in the raw data folder which is in the home directory. It also has webcam attached to laptop on which it is executed.\

### Analysis
Calculates the 10 shortest phrases by total frames by mean or median of all frames. Rest all are deleted.

### LandmarkExtraction
Extracts 258 features from both hands & torso. All features are concatenated for a frame. Total of 40 frames per video are used.\
These are stored for each phrase separatly in different directories. All the features are saved in the home directory.\
Can see actual landmark detection & tracking in real time.

### LSTM
Implements Long Short Term Memory model using Tensorflow.\
Concatenates data for all 40 frames from previously extracted features for each video ].\
Divides data in 85% training & 15% testing. TEsting data is stored as phrases & results numpy array separately in root directory\
Model has 6 layers. 3 LSTM & 3 dense layers.\
activation function: sigmoid (layers 1 to 5) softmax (last layer)\
optimizer: adem\
loss: categorical crossentropy\
metrics: cateforical accuracy\
epochs: 1200\
The above parameters are adjusted experimentally for best accuracy on testing data.\
Program logs on tensorboard to see categorical accuracy & loss function for each epoch
LSTM model is saved in root directory.

### Testing
Loads testingdata saved during LSTM.py & the LSTM model.\
Tests model on testing data.\
Generates confusion matrix & multilabel confusion matrix.

### SignLanguageTranslation
Does real time captioning.\
Loads previously detected model.\
Concatenates last 40 frames sequentially in real time for prediction.\ 
If prediction confidence is above threshhold & last 10 predictions are same then it is displayed.\
Press 'q' key to end program.
























]
