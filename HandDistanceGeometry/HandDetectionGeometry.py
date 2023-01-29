import math
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone


# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 710)

# Handdetector

detector = HandDetector(detectionCon=0.8, maxHands=2)

# where x is the distance in the image
x = [300, 245, 208, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
# where y is the depth from the camera in cm
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# The values correspond to a quadratic equation
# y = Ax^2 + Bx + c
# We can use NumPy to get the coefficients

coefficients = np.polyfit(x, y, 2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # this corresponds to the first hand that is detected
        if hands[0]:
            lmlist = hands[0]['lmList']
            p1, p2, p3, p4 = hands[0]['bbox']
            x1, y1, z1 = lmlist[5]
            x2, y2, z2 = lmlist[17]
            distance = math.sqrt((y2 - y1) ** 2 + (x1 - x2) ** 2)
            A, B, C = coefficients
            depth = A*distance**2 + B*distance + C

            cvzone .putTextRect(img, f'{int(depth)} cm', (p1, p2) )
        if len(hands) > 1:
            lmlist1 = hands[1]['lmList']
            p5, p6, p7, p8 = hands[1]['bbox']
            c1, v1, k1 = lmlist1[5]
            c2, v2, k2 = lmlist1[17]
            distance1 = math.sqrt((v2 - v1) ** 2 + (c2 - c1) ** 2)
            A, B, C = coefficients
            depth1 = A * distance1 ** 2 + B * distance1 + C
            cvzone .putTextRect(img, f'{int(depth1)} cm', (p5, p6))

    cv2.imshow('image', img)
    cv2.waitKey(1)
