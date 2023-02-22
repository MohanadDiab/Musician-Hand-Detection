import cv2
import numpy as np
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from mediapipe.python.solutions import hands

import EdgeDetection as ed


# This class is the hand detection class
# It will take a video as input
# Call the method handDetector
# This function takes the argument of the video
# put 0, 1 for a live webcam, the filename.format for a video in the file
# It defaults to 0 if no argument is provided
# the max number of hands displayed in the video are 2
# the confidence in detection is set to 0.5

class HandDetector:
    def handDetector(video=0, edges=False, flipType=True):
        # this is the tools needed to plot the nodes and lines corresponding to hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        edgeDetector = ed.EdgeDetection

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 60, (640, 480))

        # Create a black image
        image = np.zeros((480, 640, 3), np.uint8)

        # Write the black image to the video file
        for i in range(200):
            out.write(image)

        # Release the video writer
        out.release()

        black_screen = cv2.VideoCapture("output.avi")

        # cap is the actual video being taken
        # it takes as argument either the webcam or a video
        # 0 for webcam, filename.format for video
        cap = cv2.VideoCapture(video)

        # this is the hands detection object
        hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)

        # values to plot the FPS to the screen
        pTime = 0
        cTime = 0
        # these are the coordinates per frame for each node at the tip of each finger
        allHands = []
        # this while condition will work until a video is finished
        # or when the user presses escape button
        while cap.isOpened():
            success, image = cap.read()
            data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
            success, black = black_screen.read()
            # prints out on the console of not working
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            # this turns the frame into BGR instead of RGB
            # this is done since the detector needs this type of input
            # this will be changed back for the display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h, w, c = image.shape
            if results.multi_hand_landmarks:
                for hand_type, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                    myHand = {}
                    ## lmList
                    mylmList = []
                    xList = []
                    yList = []
                    zList = []

                    # this part is the part that detects the nodes location
                    # call lm.x lm.y lm.z for each node location
                    # the id is the number of the node
                    # the id is the hand landmark, 1 for each joint
                    # they sum up to 21 nodes in total
                    # the following link has a photo that displays the nodes
                    # https://google.github.io/mediapipe/solutions/hands
                    for id, lm in enumerate(hand_landmarks.landmark):
                        px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                        print(lm)
                        mylmList.append([px, py, pz])
                        xList.append(px)
                        yList.append(py)
                        zList.append(pz)

                    ## bbox
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    boxW, boxH = xmax - xmin, ymax - ymin
                    bbox = xmin, ymin, boxW, boxH
                    cx, cy = bbox[0] + (bbox[2] // 2), \
                             bbox[1] + (bbox[3] // 2)

                    myHand["lmList"] = mylmList
                    myHand["bbox"] = bbox
                    myHand["center"] = (cx, cy)

                    if flipType:
                        if hand_type.classification[0].label == "Right":
                            myHand["type"] = "Left"
                        else:
                            myHand["type"] = "Right"
                    else:
                        myHand["type"] = hand_type.classification[0].label
                    allHands.append(myHand)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    mp_drawing.draw_landmarks(
                        black,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    cv2.rectangle(image, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(image, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

            if edges:
                black = edgeDetector.drawEdges(image, black)

            # Display the FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            # this is what outputs the video
            cv2.imshow("No Background", black)
            cv2.imshow('MediaPipe Hands', image)

            # this will exit when the video ends
            # will also exit of the user presses the esc button
            if cv2.waitKey(5) & 0xFF == 27:
                break
        # this command is useful to release the camera so that it can be reused
        # otherwise it will still be operating in the background
        cap.release()
        return allHands

    def handDetectorImage(image):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9)

        image = cv2.imread(image)

        # this turns the frame into BGR instead of RGB
        # this is done since the detector needs this type of input
        # this will be changed back for the display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the hand annotations on the image.

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # this part is the part that detects the nodes location
                # call lm.x lm.y lm.z for each node location
                # the id is the number of the node
                # the id is the hand landmark, 1 for each joint
                # they sum up to 21 nodes in total
                # the following link has a photo that displays the nodes
                # https://google.github.io/mediapipe/solutions/hands
                print()
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 4:
                        z4 = lm.z
                        z = [z4]

        return z
