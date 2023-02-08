import cv2


class EdgeDetection:

    def drawEdges(frame, black):
        # Apply Canny edge detection
        edges = cv2.Canny(frame, 50, 60)

        # Draw the contours into the black screen
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(black, contours, -1, (0, 255, 0), 2)
        return black
