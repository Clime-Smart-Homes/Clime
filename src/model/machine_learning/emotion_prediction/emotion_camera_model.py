import numpy as np

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten

from keras.models import model_from_json
from keras.models import Sequential

import cv2


from model.domain.user import User


# Emotion model source:
# https://github.com/atulapra/Emotion-detection
class EmotionCameraModel:
    def __init__(self):

        # Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
        # substraction to eliminate the effect of illunination changes
        # self.model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        # Represent the 8 age classes of this CNN probability layer

        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        self.emotion_scores = {"Angry": 0, "Disgusted": 0, "Fearful": 0, "Happy": 100, "Neutral": 50, "Sad": 0, "Surprised": 100}

        self.emotions_text = [
            'angry',
            'disgust',
            'fear',
            'happy',
            'sad',
            'surprise',
            'neutral'
        ]

        self.last_emotion = ""
        self.reset_confidence = False
        # download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        face_proto = "model/machine_learning/age_prediction/weights/deploy.prototxt.txt"
        # download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
        face_model = "model/machine_learning/age_prediction/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        # Initialize frame size
        self.frame_width = 1280
        self.frame_height = 720

        emotion_proto = "model/machine_learning/emotion_prediction/weights/haarcascade_frontalface_default.xml"
        emotion_model = "model/machine_learning/emotion_prediction/weights/model.h5"
        # load face Caffe model
        self.face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
        # Load emotion prediction model
        # self.emotion_net = cv2.dnn.readNetFromCaffe(emotion_proto, emotion_model)
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        self.model.load_weights(emotion_model)

    def get_model_type(self):
        return 'emotion'

    def get_faces(self, frame, confidence_threshold=0.5):
        """Returns the box coordinates of all detected faces"""
        # convert the frame into a blob to be ready for NN input
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
        # set the image as input to the NN
        self.face_net.setInput(blob)
        # perform inference and get predictions
        output = np.squeeze(self.face_net.forward())
        # initialize the result list
        faces = []
        # Loop over the faces detected
        for i in range(output.shape[0]):
            confidence = output[i, 2]
            if confidence > confidence_threshold:
                box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(np.int)
                # widen the box a little
                start_x, start_y, end_x, end_y = start_x - \
                                                 10, start_y - 10, end_x + 10, end_y + 10
                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = 0 if end_x < 0 else end_x
                end_y = 0 if end_y < 0 else end_y
                # append to our list
                faces.append((start_x, start_y, end_x, end_y))
        return faces

    def predict(self, frame):
        faces = self.get_faces(frame)  # TODO: Determine order of faces for polishing
        if len(faces) == 0:
            return None, None, None, None

        (start_x, start_y, end_x, end_y) = faces[0]
        user = User(start_x, start_y, end_x, end_y)

        face_img = frame[user.start_y: user.end_y, user.start_x: user.end_x]

        face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
        # image --> Input image to preprocess before passing it through our dnn for classification.
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(face_img, (48, 48)), -1), 0)
        # Predict emotion
        prediction = self.model.predict(cropped_img)
        
        maxindex = int(np.argmax(prediction))
        emotion_detected = self.emotion_dict[maxindex]

        self.last_emotion = emotion_detected

        model_info = f'Emotion detected: {emotion_detected}'
        # cv2.putText(frame, emotion_detected, (user.start_x+20, user.start_y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        output_value = self.emotion_scores[emotion_detected]
        confidence_score = prediction[0][maxindex] 

        return user, output_value, confidence_score, model_info
