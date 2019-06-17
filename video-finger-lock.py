import cv2
import numpy as np
from math import floor
import time
import os

# Password - change here to expect a different one
password = [("Top-Left", 0), ("Center", 2), ("Top-Right", 5), ("Center", 3), ("Bottom-Center", 0)]

# Threshold values - can be tuned as light changes in the environment
redThreshold = 150
grayConversionThreshold = 60
countedContourSize = 50
circleRadiusAdjustment = 0.38

# Edge values, can be tuned to require more or less hand movement in order
# to change the hand location
topEdge = 250
bottomEdge = 500
leftEdge = 425
rightEdge = 850

# Gets the string corresponding to where the hand is based on x, y position
def getLoc(x, y):
    if x < leftEdge:
        if y < topEdge:
            return "Top-Left"
        elif y > bottomEdge:
            return "Bottom-Left"
        else:
            return "Center-Left"
    elif x > rightEdge:
        if y < topEdge:
            return "Top-Right"
        elif y > bottomEdge:
            return "Bottom-Right"
        else:
            return "Center-Right"
    else:
        if y < topEdge:
            return "Top-Center"
        elif y > bottomEdge:
            return "Bottom-Center"
        else:
            return "Center"            

if __name__ == "__main__":
    # Initialize video capture and output directory
    camera = cv2.VideoCapture(0)
    attemptTime = str(time.time())
    dirName = "./examples/attempt-" + attemptTime
    os.mkdir(dirName)
    count = 0
    passwordAttempt = []
    while(True):
        # Get initial camera frame
        couldRead, initialFrame = camera.read()
        frame = cv2.flip(initialFrame, 1).copy()
        # Create a new frame where all pixels over a certain red threshold
        # will be white and all other pixels will be black
        redAsWhiteFrame = frame.copy()
        redOverlay = np.where(redAsWhiteFrame[:, :, 2] > redThreshold, 255, 0)
        redAsWhiteFrame[:, :, 0] = redOverlay
        redAsWhiteFrame[:, :, 1] = redOverlay
        redAsWhiteFrame[:, :, 2] = redOverlay
        # Lay out text on both original frame and new frame
        cv2.putText(redAsWhiteFrame, "Image combo count: " + str(count), (25, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(redAsWhiteFrame, "Press 'f' to capture image", (25, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(redAsWhiteFrame, "Press 'j' to submit image combo", (25, 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, "Image combo count: " + str(count), (25, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, "Press 'f' to capture image", (25, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, "Press 'j' to submit image combo", (25, 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)

        # Make a binary map using the new frame
        redBinaryFrame = frame.copy()
        redBinaryFrame[:, :, 2] = np.where(redAsWhiteFrame[:, :, 2] > redThreshold, 1, 0)
        # Find the contours in the binary map
        contours, hierarchy = cv2.findContours(redBinaryFrame[:, :, 2], 1, 2)
        if len(contours) > 0:
            # Identify the hand as the largest contour (due to domain engineering)
            contour = max(contours, key = cv2.contourArea)
            M = cv2.moments(contour)
            # Draw a bounding rectangle
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(redAsWhiteFrame,(x,y),(x+w,y+h),(0,255,0),2)
            # Create circle for finger counting
            circleRadius = int(floor(max(w, h) * circleRadiusAdjustment))
            circleCircumference = 2 * np.pi * circleRadius
            centerX = x + (w / 2)
            centerY = y + (h / 2)
            # Draw center point of hand
            cv2.circle(redAsWhiteFrame, (centerX, centerY), 10, (0, 0, 255), -1)
            # Draw circle around hand to be displayed
            cv2.circle(redAsWhiteFrame, (centerX, centerY), circleRadius, (0, 255, 0),2)
            # Draw circular mask
            circular_mask = np.zeros(redBinaryFrame.shape[:2], dtype="uint8")
            cv2.circle(circular_mask, (centerX, centerY), circleRadius, (255, 255, 255), 2)
            # Perform bitwise and between the circular mask and the binary frame
            circular_and = cv2.bitwise_and(redBinaryFrame, redBinaryFrame, mask=circular_mask)
            # Find all the identified finger contours
            _, circular_and_threshold = cv2.threshold(cv2.cvtColor(circular_and,cv2.COLOR_BGR2GRAY), grayConversionThreshold, 255, 0)
            fingerContours, _ = cv2.findContours(circular_and_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Count up all fingers
            fingerContourCount = -1 # start at -1 to account for wrist
            for cntr in fingerContours:
                cntrArea = cv2.contourArea(cntr)
                # Only counts if bigger than some area (not noise) and
                # smaller than 1/4 of the circle (not section of hand)
                if cntrArea > countedContourSize and cntrArea < circleCircumference * 0.25:
                    fingerContourCount += 1
            if fingerContourCount == -1: # if wrist too large, might fall below 0 -> shouldn't be below 0
                fingerContourCount = 0
            # Display number found, location found, and draw additional metadata for machine display/debugging
            cv2.putText(redAsWhiteFrame, "Number found: " + str(fingerContourCount), (25, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
            cv2.putText(frame, "Number found: " + str(fingerContourCount), (25, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
            cv2.line(redAsWhiteFrame, (leftEdge, 0), (leftEdge, 2000), (0, 0, 255), 1)
            cv2.line(redAsWhiteFrame, (rightEdge, 0), (rightEdge, 2000), (0, 0, 255), 1)
            cv2.line(redAsWhiteFrame, (0, topEdge), (2000, topEdge), (0, 0, 255), 1)
            cv2.line(redAsWhiteFrame, (0, bottomEdge), (2000, bottomEdge), (0, 0, 255), 1)
            locFound = getLoc(centerX, centerY)
            cv2.putText(redAsWhiteFrame, "Location found: " + locFound, (25, 250), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
            cv2.putText(frame, "Location found: " + locFound, (25, 250), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
            cv2.drawContours(redAsWhiteFrame, fingerContours, -1, (0, 0, 255), 3)

        # Display video feed and wait for key capture
        cv2.imshow("Video Feed", frame) # Change frame to redAsWhiteFrame to see what the computer sees
        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('q'): # If q pressed - exit program
            break
        elif pressedKey == ord('f'): # If f pressed - record current value and write images
            count += 1
            passwordAttempt.append((locFound, fingerContourCount))
            cv2.imwrite(dirName + "/cam-view-" + str(count) + ".jpg", frame)
            cv2.imwrite(dirName + "/mach-view-" + str(count) + ".jpg", redAsWhiteFrame)
        elif pressedKey == ord('j'): # If j - "submit" password and right if it was correct
            f = open(dirName + "/results.txt", "w")
            f.write("Correct password: " + str(password) + "\n")
            f.write("Attempted password: " + str(passwordAttempt) + "\n")
            if passwordAttempt == password:
                f.write("Correct Password")
                print("Correct Password")
            else:
                f.write("Incorrect Password")
                print("Incorrect Password")
            break
