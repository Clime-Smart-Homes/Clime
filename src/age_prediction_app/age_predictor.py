import cv2
import numpy as np
from model.user import User

class AgePredictor:
    def __init__(self):

        # The model architecture
        # download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
        age_model = 'age_prediction_app/weights/deploy_age.prototxt'
        # Thedel pre-trained weights
        # download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
        age_proto = 'age_prediction_app/weights/age_net.caffemodel'
        # Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
        # substraction to eliminate the effect of illunination changes
        self.model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        # Represent the 8 age classes of this CNN probability layer
        self.age_intervals = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                         '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

        self.age_trust = {
            '(0, 2)': 15,
            '(4, 6)': 25,
            '(8, 12)': 35,
            '(15, 20)': 65,
            '(25, 32)': 85,
            '(38, 43)': 100,
            '(48, 53)': 100,
            '(60, 100)': 35
        }
        # download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        face_proto = "age_prediction_app/weights/deploy.prototxt.txt"
        # download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
        face_model = "age_prediction_app/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        # Initialize frame size
        self.frame_width = 1280
        self.frame_height = 720

        # load face Caffe model
        self.face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
        # Load age prediction model
        self.age_net = cv2.dnn.readNetFromCaffe(age_model, age_proto)

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

    def get_optimal_font_scale(self, text, width):
        """Determine the optimal font scale based on the hosting frame width"""
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale / 10
        return 1

    # from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        # resize the image
        return cv2.resize(image, dim, interpolation=inter)

    def predict_age(self, frame):
        faces = self.get_faces(frame)  # TODO: Determine order of faces for polishing
        if len(faces) == 0:
            return None, None, None

        (start_x, start_y, end_x, end_y) = faces[0]
        user = User(start_x, start_y, end_x, end_y)
        
        face_img = frame[user.start_y: user.end_y, user.start_x: user.end_x]

        # image --> Input image to preprocess before passing it through our dnn for classification.
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=self.model_mean_values, swapRB=False
        )
        # Predict Age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        print("=" * 30, "Face Prediction Probabilities", "=" * 30)
        for i in range(age_preds[0].shape[0]):
            print(f"{self.age_intervals[i]}: {age_preds[0, i] * 100:.2f}%")
        i = age_preds[0].argmax()
        age = self.age_intervals[i]
        age_confidence_score = age_preds[0][i]

        return user, self.age_trust[age], age_confidence_score

