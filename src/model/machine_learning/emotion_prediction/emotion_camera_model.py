import cv2
import numpy as np

from model.domain.user import User


class EmotionCameraModel:
    def __init__(self):

        # Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
        # substraction to eliminate the effect of illunination changes
        self.model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        # Represent the 8 age classes of this CNN probability layer

        self.emotions_text = [
            'angry',
            'disgust',
            'fear',
            'happy',
            'sad',
            'surprise',
            'neutral'
        ]
        # download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        face_proto = "model/machine_learning/age_prediction/weights/deploy.prototxt.txt"
        # download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
        face_model = "model/machine_learning/age_prediction/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        # Initialize frame size
        self.frame_width = 1280
        self.frame_height = 720

        emotion_proto = "model/machine_learning/emotion_prediction/weights/emotion_miniXception.prototxt.txt"
        emotion_model = "model/machine_learning/emotion_prediction/weights/emotion_miniXception.caffemodel"
        # load face Caffe model
        self.face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
        # Load emotion prediction model
        self.emotion_net = cv2.dnn.readNet(emotion_proto, emotion_model)

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

    def predict_age(self, frame):
        faces = self.get_faces(frame)  # TODO: Determine order of faces for polishing
        if len(faces) == 0:
            return None, None, None

        (start_x, start_y, end_x, end_y) = faces[0]
        user = User(start_x, start_y, end_x, end_y)

        face_img = frame[user.start_y: user.end_y, user.start_x: user.end_x]
        print("X: (" + str(user.start_x) + ", " + str(user.end_x) + ")")
        print("y: (" + str(user.start_y) + ", " + str(user.end_y) + ")")

        face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
        # image --> Input image to preprocess before passing it through our dnn for classification.
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(64, 64),
            swapRB=False
        )
        # Predict emotion
        self.emotion_net.setInput(blob)
        emotion_preds = self.emotion_net.forward()
        print("=" * 30, "Emotion Prediction Probabilities", "=" * 30)
        print(emotion_preds)
        # for i in range(emotion_preds[0].shape[0]):
            #print(f"{self.age_intervals[i]}: {age_preds[0, i] * 100:.2f}%")
            # print(f"{emotion_preds[0, i] * 100:.2f}%")
        i = emotion_preds.argmax()
        print("i = " + str(i))
        if i != 0 and i != 6:
            print("WE DID IT, NTRO ANGRY AND NOT NETURAL")
        print(self.emotions_text[i])
        #age = self.age_intervals[i]
        emotion_confidence_score = emotion_preds[0][i]

        return user, 10, emotion_confidence_score
