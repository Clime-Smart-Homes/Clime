from utils.drawing_utils import *
from model.machine_learning.age_prediction.age_camera_model import AgeCameraModel
from model.machine_learning.emotion_prediction.emotion_camera_model import EmotionCameraModel

import cv2
import os

AVG_WINDOW_LEN = 5

class App():
    def __init__(self):
        self.age_model = AgeCameraModel()
        self.emotion_model = EmotionCameraModel()

        self.can_display = self.has_graphics()

        self.current_model = self.age_model
        
    def has_graphics(self):
        if 'XDG_SESSION_TYPE' in os.environ:
            return True

        return False

    def switch_model(self, model_name):
        if model_name == 'age':
            self.current_model = self.age_model

        elif model_name == 'emotion':
            self.current_model = self.emotion_model

    def process_images(self):
        """Predict the age of the faces showing in the image"""
        confidence = 0
        output = 0

        os.chdir('/home/hibban/Downloads/Clime/src/images')
        img_num = 0
        camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

        avg = []
        tm = cv2.TickMeter()
        print("Beginning capture")
        while True:
            tm.start()
            _, img = camera.read()
            # Take a copy of the initial image and resize it
            frame = img

            if frame is None:
                raise Exception("No camera detected")

            if frame.shape[1] > self.current_model.frame_width:
                frame = image_resize(frame, width=self.current_model.frame_width)

            user, value, pred_confidence, reduce_confidence = self.current_model.predict(frame)

            tm.stop()
            fps= f'{tm.getFPS():.2f}'
    

            if reduce_confidence:
                confidence -= 10
                output -= 5

            # Slowly decrease confidence and output value if user is not found
            elif user is None:
                confidence -= 10
                output -= 10

                if confidence < 0:
                    confidence = 0
                if output < 0:
                    output = 0

            else:
                if pred_confidence >= confidence:
                    confidence = pred_confidence
                    output = round((value + output) / 2) + 1

                label = "Controller Value: " + str(value) + " -- Output value: " + str(output) + " -- FPS: " + str(fps)

                draw_box(frame, user, label)

            # Display processed image
            if self.can_display:
                img_num += 1
                num_str = str(img_num).zfill(6)
                save_name = f"img_{num_str}.jpeg"
                # Uncomment to save imsages
                # cv2.imwrite(save_name, frame)
                cv2.imshow('Smart Faucet', frame)
            
            avg.append(output)

            if len(avg) > AVG_WINDOW_LEN:
                avg.pop(0)

            if len(avg) > 0:
                yield sum(avg) / len(avg)
            else:
                yield 0

            if cv2.waitKey(1) == ord("q"):
                break

            tm.reset()


        camera.release()
        cv2.destroyAllWindows()
