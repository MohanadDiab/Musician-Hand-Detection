import cv2
import numpy as np
import mediapipe as mp
import time
import matplotlib.pyplot as plt


# This class is the hand detection class
# It will take a video as input
# Call the method handDetector
# This function takes the argument of the video
# put 0, 1 for a live webcam, the filename.format for a video in the file
# It defaults to 0 if no argument is provided
# the max number of hands displayed in the video are 2
# the confidence in detection is set to 0.5

class HandDetector():
    def handDetector(video=0):
        # this is the tools needed to plot the nodes and lines corresponding to hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        # cap is the actual video being taken
        # it takes as argument either the webcam or a video
        # 0 for webcam, filename.format for video
        cap = cv2.VideoCapture(video)

        # this is the hands detection object
        hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # values to plot the FPS to the screen
        pTime = 0
        cTime = 0
        # these are the coordinates per frame for each node at the tip of each finger
        z4 = []
        z8 = []
        z12 = []
        z16 = []
        z20 = []

        # this while condition will work until a video is finished
        # or when the user presses escape button
        while cap.isOpened():
            success, image = cap.read()

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
                    for id, lm in enumerate(hand_landmarks.landmark):
                        if id == 4:
                            print(lm)
                            z4.append(lm.x)
                        if id == 8:
                            z8.append(lm.x)
                        if id == 12:
                            z12.append(lm.x)
                        if id == 16:
                            z16.append(lm.x)
                        if id == 20:
                            z20.append(lm.x)

            # Display the FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            # this is what outputs the video
            cv2.imshow('MediaPipe Hands', image)

            # this will exit when the video ends
            # will also exit of the user presses the esc button
            if cv2.waitKey(5) & 0xFF == 27:
                break
        # this command is useful to release the camera so that it can be reused
        # otherwise it will still be operating in the background
        cap.release()

        # Plotting
        # Plot the data for the thump, id = 4
        plt.plot(z4)
        # Set the y-axis label
        plt.ylabel('Thump depth')
        # Show the plot
        plt.show()

        # Plot the data for the index, id = 8
        plt.plot(z8)
        # Set the y-axis label
        plt.ylabel('Index depth')
        # Show the plot
        plt.show()

        # Plot the data for the Middle, id = 12
        plt.plot(z12)
        # Set the y-axis label
        plt.ylabel('Middle depth')
        # Show the plot
        plt.show()

        # Plot the data for the Ring, id = 16
        plt.plot(z16)
        # Set the y-axis label
        plt.ylabel('Ring depth')
        # Show the plot
        plt.show()

        # Plot the data for the Pinky, id = 20
        plt.plot(z16)
        # Set the y-axis label
        plt.ylabel('Pinky depth')
        # Show the plot
        plt.show()
