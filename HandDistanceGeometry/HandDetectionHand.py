import math
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
from datetime import datetime

# Webcam

cap = cv2.VideoCapture(0)

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
hand1_entries = []
hand2_entries = []

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # this corresponds to the first hand that is detected
        if hands[0]:
            # Getting the landmark list
            lmlist = hands[0]['lmList']

            # Getting the bounding box of the hand
            p1, p2, p3, p4 = hands[0]['bbox']

            # The coordinates of the joints
            x0, y0, z0 = lmlist[0]
            x1, y1, z1 = lmlist[1]
            x2, y2, z2 = lmlist[2]
            x3, y3, z3 = lmlist[3]
            x4, y4, z4 = lmlist[4]
            x5, y5, z5 = lmlist[5]
            x6, y6, z6 = lmlist[6]
            x7, y7, z7 = lmlist[7]
            x8, y8, z8 = lmlist[8]
            x9, y9, z9 = lmlist[9]
            x10, y10, z10 = lmlist[10]
            x11, y11, z11 = lmlist[11]
            x12, y12, z12 = lmlist[12]
            x13, y13, z13 = lmlist[13]
            x14, y14, z14 = lmlist[14]
            x15, y15, z15 = lmlist[15]
            x16, y16, z16 = lmlist[16]
            x17, y17, z17 = lmlist[17]
            x18, y18, z18 = lmlist[18]
            x19, y19, z19 = lmlist[19]
            x20, y20, z20 = lmlist[20]

            # estimation of the palm plane depth
            distanceBase1 = math.sqrt((y5 - y17) ** 2 + (x5 - x17) ** 2)
            A, B, C = coefficients
            depth1 = A * distanceBase1 ** 2 + B * distanceBase1 + C

            # determination of the ratio of the line segments
            # the line segments represent the lines joining the articulations

            # for the thump
            distance_4_3 = math.sqrt((y4 - y3) ** 2 + (x4 - x3) ** 2)
            distance_3_2 = math.sqrt((y3 - y2) ** 2 + (x3 - x2) ** 2)

            # for the index
            distance_6_7 = math.sqrt((y6 - y7) ** 2 + (x6 - x7) ** 2)
            distance_7_8 = math.sqrt((y8 - y7) ** 2 + (x8 - x7) ** 2)

            # for the middle finger
            distance_12_11 = math.sqrt((y12 - y11) ** 2 + (x12 - x11) ** 2)
            distance_11_10 = math.sqrt((y11 - y10) ** 2 + (x11 - x10) ** 2)

            # for the ring finger
            distance_16_15 = math.sqrt((y15 - y16) ** 2 + (x16 - x15) ** 2)
            distance_15_14 = math.sqrt((y15 - y14) ** 2 + (x15 - x14) ** 2)

            # for the pinky finger
            distance_20_19 = math.sqrt((y20 - y19) ** 2 + (x20 - x19) ** 2)
            distance_19_18 = math.sqrt((y19 - y18) ** 2 + (x19 - x18) ** 2)

            # Average finger length
            thump_length1 = 5.0
            index_length1 = 6.7
            middle_length1 = 8.8
            ring_length1 = 6.8
            pinky_length1 = 5.7

            # ratios and depths estimation
            thump_ratio1 = distance_4_3 / distance_3_2
            index_ratio1 = distance_7_8 / distance_6_7
            middle_ratio1 = distance_12_11 / distance_11_10
            ring_ratio1 = distance_16_15 / distance_15_14
            pinky_ratio1 = distance_20_19 / distance_19_18

            joint0_depth1 = depth1
            joint1_depth1 = depth1
            joint2_depth1 = depth1
            joint3_depth1 = thump_length1 * (0.9 - thump_ratio1) / 2 + depth1
            joint4_depth1 = thump_length1 * (0.9 - thump_ratio1) + depth1
            joint5_depth1 = depth1
            joint6_depth1 = depth1
            joint7_depth1 = index_length1 * (0.9 - index_ratio1) / 2 + depth1
            joint8_depth1 = index_length1 * (0.9 - index_ratio1) + depth1
            joint9_depth1 = depth1
            joint10_depth1 = depth1
            joint11_depth1 = middle_length1 * (0.9 - middle_ratio1) / 2 + depth1
            joint12_depth1 = middle_length1 * (0.9 - middle_ratio1) + depth1
            joint13_depth1 = depth1
            joint14_depth1 = depth1
            joint15_depth1 = ring_length1 * (0.9 - ring_ratio1) / 2 + depth1
            joint16_depth1 = ring_length1 * (0.9 - ring_ratio1) + depth1
            joint17_depth1 = depth1
            joint18_depth1 = depth1
            joint19_depth1 = pinky_length1 * (0.9 - pinky_ratio1) / 2 + depth1
            joint20_depth1 = pinky_length1 * (0.9 - pinky_ratio1) + depth1
            joint_depths1 = [
                joint0_depth1,
                joint1_depth1,
                joint2_depth1,
                joint3_depth1,
                joint4_depth1,
                joint5_depth1,
                joint6_depth1,
                joint7_depth1,
                joint8_depth1,
                joint9_depth1,
                joint10_depth1,
                joint11_depth1,
                joint12_depth1,
                joint13_depth1,
                joint14_depth1,
                joint15_depth1,
                joint16_depth1,
                joint17_depth1,
                joint18_depth1,
                joint19_depth1,
                joint20_depth1,
            ]
            hand1_entries.append(joint_depths1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'{int(joint8_depth1)} cm', (x8, y8), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f'{int(joint7_depth1)} cm', (x7, y7), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f'{int(depth1)} cm', (x6, y6), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cvzone.putTextRect(img, f'{int(depth1)} cm', (p1, p2))

            # Now the second hand
            if len(hands) > 1:
                lmlist1 = hands[1]['lmList']
                p5, p6, p7, p8 = hands[1]['bbox']

                # The coordinates of the joints
                c0, v0, k0 = lmlist1[0]
                c1, v1, k1 = lmlist1[1]
                c2, v2, k2 = lmlist1[2]
                c3, v3, k3 = lmlist1[3]
                c4, v4, k4 = lmlist1[4]
                c5, v5, k5 = lmlist1[5]
                c6, v6, k6 = lmlist1[6]
                c7, v7, k7 = lmlist1[7]
                c8, v8, k8 = lmlist1[8]
                c9, v9, k9 = lmlist1[9]
                c10, v10, k10 = lmlist1[10]
                c11, v11, k11 = lmlist1[11]
                c12, v12, k12 = lmlist1[12]
                c13, v13, k13 = lmlist1[13]
                c14, v14, k14 = lmlist1[14]
                c15, v15, k15 = lmlist1[15]
                c16, v16, k16 = lmlist1[16]
                c17, v17, k17 = lmlist1[17]
                c18, v18, k18 = lmlist1[18]
                c19, v19, k19 = lmlist1[19]
                c20, v20, k20 = lmlist1[20]

                # estimation of the palm plane depth
                distanceBase2 = math.sqrt((v5 - v17) ** 2 + (c5 - c17) ** 2)
                A, B, C = coefficients
                depth2 = A * distanceBase2 ** 2 + B * distanceBase2 + C

                # determination of the ratio of the line segments
                # the line segments represent the lines joining the articulations

                # for the thump
                distance2_4_3 = math.sqrt((v4 - v3) ** 2 + (c4 - c3) ** 2)
                distance2_3_2 = math.sqrt((v3 - v2) ** 2 + (c3 - c2) ** 2)

                # for the index
                distance2_6_7 = math.sqrt((v6 - v7) ** 2 + (c6 - c7) ** 2)
                distance2_7_8 = math.sqrt((v8 - v7) ** 2 + (c8 - c7) ** 2)

                # for the middle finger
                distance2_12_11 = math.sqrt((v12 - v11) ** 2 + (c12 - c11) ** 2)
                distance2_11_10 = math.sqrt((v11 - v10) ** 2 + (c11 - c10) ** 2)

                # for the ring finger
                distance2_16_15 = math.sqrt((v15 - v16) ** 2 + (c16 - c15) ** 2)
                distance2_15_14 = math.sqrt((v15 - v14) ** 2 + (c15 - c14) ** 2)

                # for the pinky finger
                distance2_20_19 = math.sqrt((v20 - v19) ** 2 + (c20 - c19) ** 2)
                distance2_19_18 = math.sqrt((v19 - v18) ** 2 + (c19 - c18) ** 2)

                # Average finger length
                thump_length1 = 5.0
                index_length1 = 6.7
                middle_length1 = 8.8
                ring_length1 = 6.8
                pinky_length1 = 5.7

                # ratios and depths estimation
                thump_ratio2 = distance2_4_3 / distance2_3_2
                index_ratio2 = distance2_7_8 / distance2_6_7
                middle_ratio2 = distance2_12_11 / distance2_11_10
                ring_ratio2 = distance2_16_15 / distance2_15_14
                pinky_ratio2 = distance2_20_19 / distance2_19_18

                joint0_depth2 = depth2
                joint1_depth2 = depth2
                joint2_depth2 = depth2
                joint3_depth2 = thump_length1 * (0.9 - thump_ratio2) / 2 + depth2
                joint4_depth2 = thump_length1 * (0.9 - thump_ratio2) + depth2
                joint5_depth2 = depth2
                joint6_depth2 = depth2
                joint7_depth2 = index_length1 * (0.9 - index_ratio2) / 2 + depth2
                joint8_depth2 = index_length1 * (0.9 - index_ratio2) + depth2
                joint9_depth2 = depth2
                joint10_depth2 = depth2
                joint11_depth2 = middle_length1 * (0.9 - middle_ratio2) / 2 + depth2
                joint12_depth2 = middle_length1 * (0.9 - middle_ratio2) + depth2
                joint13_depth2 = depth2
                joint14_depth2 = depth2
                joint15_depth2 = ring_length1 * (0.9 - ring_ratio2) / 2 + depth2
                joint16_depth2 = ring_length1 * (0.9 - ring_ratio2) + depth2
                joint17_depth2 = depth2
                joint18_depth2 = depth2
                joint19_depth2 = pinky_length1 * (0.9 - pinky_ratio2) / 2 + depth2
                joint20_depth2 = pinky_length1 * (0.9 - pinky_ratio2) + depth2
                joint_depths2 = [
                    joint0_depth2,
                    joint1_depth2,
                    joint2_depth2,
                    joint3_depth2,
                    joint4_depth2,
                    joint5_depth2,
                    joint6_depth2,
                    joint7_depth2,
                    joint8_depth2,
                    joint9_depth2,
                    joint10_depth2,
                    joint11_depth2,
                    joint12_depth2,
                    joint13_depth2,
                    joint14_depth2,
                    joint15_depth2,
                    joint16_depth2,
                    joint17_depth2,
                    joint18_depth2,
                    joint19_depth2,
                    joint20_depth2,
                ]
                hand2_entries.append(joint_depths2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, f'{int(joint8_depth2)} cm', (c8, v8), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f'{int(joint7_depth2)} cm', (c7, v7), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f'{int(depth2)} cm', (c6, v6), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cvzone.putTextRect(img, f'{int(depth2)} cm', (p5, p6))

    cv2.imshow('image', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = 'articulations_' + timestamp + '.txt'
with open(file_name, 'w') as file:
    # Write each item in the list as a separate line
    file.write("First hand entries\n")
    hand1_entries = np.array(hand1_entries)
    hand2_entries = np.array(hand2_entries)
    for sublist in hand1_entries:
        file.write(", ".join(str(x) for x in sublist) + "\n")

    file.write("Second hand entries\n")
    for sublist in hand2_entries:
        file.write(", ".join(str(x) for x in sublist) + "\n")
