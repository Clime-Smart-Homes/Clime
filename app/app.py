import cv2
from AgePredictor import AgePredictor

from pyControl4.account import C4Account
from pyControl4.director import C4Director
from pyControl4.light import C4Light
import asyncio
import json
import socket

"""Put Control4 account username and password here"""
username = ""
password = ""

ip = "192.168.0.13"

"""Authenticate with Control4 account"""
account = C4Account(username, password)
asyncio.run(account.getAccountBearerToken())

"""Get and print controller name"""
accountControllers = asyncio.run(account.getAccountControllers())
print(accountControllers["controllerCommonName"])

"""Get bearer token to communicate with controller locally"""
director_bearer_token = asyncio.run(
            account.getDirectorBearerToken(accountControllers["controllerCommonName"])
            )["token"]

"""Create new C4Director instance"""
director = C4Director(ip, director_bearer_token)

"""Create new C4Light instance, put your own device for the ID"""
light = C4Light(director, 31)

camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
model = AgePredictor()

"""Predict the age of the faces showing in the image"""
confidence = 0
output = 0
while True:
    _, img = camera.read()
    # Take a copy of the initial image and resize it
    frame = img.copy()

    if frame.shape[1] > model.frame_width:
        frame = model.image_resize(frame, width=model.frame_width) #TODO create image manipulation utils module

    user, value, pred_confidence = model.predict_age(frame)

    label = "Controller Value: " + str(value) + " -- Camera value: " + str(output)
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

        # TODO: Make draw box part of image utils
        yPos = user[1] - 15 #TODO make user an object
        while yPos < 15:
            yPos += 15
        # write the text into the frame
        cv2.putText(frame, label, (user[0], yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
        # draw the rectangle around the face
        cv2.rectangle(frame, (user[0], user[1]), (user[2], user[3]), color=(255, 0, 0), thickness=2)

    # Display processed image
    cv2.imshow('Age Estimator', frame)

    yield output
    # TODO: Move network call to controller
    asyncio.run(light.rampToLevel(output, 100)) # Send value to Control4 Director

    if cv2.waitKey(1) == ord("q"):
        break
    # save the image if you want
    # cv2.imwrite("predicted_age.jpg", frame)
cv2.destroyAllWindows()

