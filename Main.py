import cv2
import numpy as np
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import HandTrackingModule as htm

#########  PLACE YOUR VIDEO IN THE VARIABLE VIDEO #######

# e.g Video = 'monkey-dancing.mp4'
#video = 'piano.mp4'
#video = 'accordion.mp4'
#video = 'guitar.mp4'
#video = 0;

#########  KEEP EQUAL TO ZERO IN CASE OF WEBCAM ########

# Create an object of the HandDetector class

handDetector = htm.HandDetector

# Call the method handDetector
# This function takes the argument of the video
# put 0, 1 for a live webcam, the filename.format for a video in the file
# It defaults to 0 if no argument is provided

handDetector.handDetector(video)
