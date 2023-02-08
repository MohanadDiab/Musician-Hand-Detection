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
            x5, y5, z5 = lmlist[5]
            x6, y6, z6 = lmlist[6]
            x7, y7, z7 = lmlist[7]
            x8, y8, z8 = lmlist[8]
            x17, y17, z17 = lmlist[17]
            distanceBase = math.sqrt((y5 - y17) ** 2 + (x5 - x17) ** 2)
            A, B, C = coefficients
            depth = A*distanceBase**2 + B*distanceBase + C

            distance_6_7 = math.sqrt((y6 - y7) ** 2 + (x6 - x7) ** 2)
            distance_7_8 = math.sqrt((y8 - y7) ** 2 + (x8 - x7) ** 2)

            index_length = 6.7
            index_ratio = distance_7_8 / distance_6_7
            max_depth = index_length * (0.9 - index_ratio) + depth
            half_depth = index_length * (0.9 - index_ratio) / 2 + depth

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'Joint 8 depth = {max_depth} cm', (10, 25), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f'Joint 7 depth = {half_depth} cm', (10, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f'Joint 6 depth = {depth} cm', (10, 95), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print(index_ratio)
            cvzone .putTextRect(img, f'{int(depth)} cm', (p1, p2))

    cv2.imshow('image', img)
    cv2.waitKey(1)
