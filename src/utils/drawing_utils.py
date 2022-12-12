import cv2


def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale / 10
    return 1


def draw_box(frame, user, label, model_info):
    yPos = user.start_y - 15  
    while yPos < 15:
        yPos += 15
    # write the text into the frame
    cv2.putText(frame, label, (user.start_x, yPos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

    cv2.putText(frame, model_info, (user.start_x, user.end_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
    # draw the rectangle around the face
    cv2.rectangle(frame, (user.start_x, user.start_y), (user.end_x, user.end_y), color=(255, 0, 0), thickness=2)
    return frame


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

