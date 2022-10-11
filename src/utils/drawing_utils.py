import cv2
from model import user

class DrawingUtils:
    def draw_box(frame, user, label):
        yPos = user.start_y - 15 #TODO make user an object
        while yPos < 15:
            yPos += 15
        # write the text into the frame
        cv2.putText(frame, label, (user.start_x, yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
        # draw the rectangle around the face
        cv2.rectangle(frame, (user.start_x, user.start_y), (user.end_x, user.end_y), color=(255, 0, 0), thickness=2)
        return frame