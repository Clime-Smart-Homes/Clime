import cv2
from model.user import *
from utils.drawing_utils import *
from age_prediction_app.age_predictor import AgePredictor

def process_images():
    """Predict the age of the faces showing in the image"""
    confidence = 0
    output = 0
    camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    model = AgePredictor()
    while True:
        _, img = camera.read()
        # Take a copy of the initial image and resize it
        frame = img.copy()

        if frame.shape[1] > model.frame_width:
            frame = model.image_resize(frame, width=model.frame_width) #TODO create image manipulation utils module

        user, value, pred_confidence = model.predict_age(frame)

        label = "Controller Value: " + str(value) + " -- Output value: " + str(output)
        print(label)

        # Slowly decrease confidence and output value if user is not found
        if user is None:
            confidence -= 1 
            output -= 1 

            if confidence < 0:
                confidence = 0
            if output < 0:
                output = 0

        else:
            if pred_confidence > confidence:
                confidence = pred_confidence
                output = value
            DrawingUtils.draw_box(frame, user, label)


        # Display processed image
        cv2.imshow('Age Estimator', frame)

        yield output

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()