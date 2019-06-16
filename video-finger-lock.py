import cv2
import numpy as np
from math import floor
import time
import os

password = [("Center", 0), ("Top-Right", 5)]

redThreshold = 150
grayConversionThreshold = 70
countedContourSize = 50

topEdge = 250
bottomEdge = 500
leftEdge = 425
rightEdge = 850

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
    camera = cv2.VideoCapture(0)
    attemptTime = str(time.time())
    dirName = "./examples/attempt-" + attemptTime
    os.mkdir(dirName)
    count = 0
    passwordAttempt = []
    while(True):
        couldRead, initialFrame = camera.read()
        frame = cv2.flip(initialFrame, 1).copy()
        redAsWhiteFrame = frame.copy()
        redOverlay = np.where(redAsWhiteFrame[:, :, 2] > redThreshold, 255, 0)
        redAsWhiteFrame[:, :, 0] = redOverlay
        redAsWhiteFrame[:, :, 1] = redOverlay
        redAsWhiteFrame[:, :, 2] = redOverlay
        cv2.putText(redAsWhiteFrame, "Image combo count: " + str(count), (25, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(redAsWhiteFrame, "Press 'f' to capture image", (25, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(redAsWhiteFrame, "Press 'j' to submit image combo", (25, 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, "Image combo count: " + str(count), (25, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, "Press 'f' to capture image", (25, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, "Press 'j' to submit image combo", (25, 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)

        redBinaryFrame = frame.copy()
        redBinaryFrame[:, :, 2] = np.where(redAsWhiteFrame[:, :, 2] > redThreshold, 1, 0)
        contours,hierarchy = cv2.findContours(redBinaryFrame[:,:,2], 1, 2)
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            M = cv2.moments(contour)
            if (M['m00'] != 0):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(redAsWhiteFrame, (cx, cy), 10, (0,0,255), -1)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(redAsWhiteFrame,(x,y),(x+w,y+h),(0,255,0),2)
            circleRadius = int(floor(max(w, h) * .38))
            circleCircumference = 2 * np.pi * circleRadius
            circular_mask = np.zeros(redBinaryFrame.shape[:2], dtype="uint8")
            centerX = x + (w / 2)
            centerY = y + (h / 2)
            cv2.circle(circular_mask, (centerX, centerY), circleRadius, (255,255,255), 2) # draw circular mask
            cv2.circle(redAsWhiteFrame, (centerX, centerY), circleRadius, (0,255,0),2) # draw circle to be displayed
            circular_and = cv2.bitwise_and(redBinaryFrame, redBinaryFrame, mask=circular_mask)
            _, circular_and_threshold = cv2.threshold(cv2.cvtColor(circular_and,cv2.COLOR_BGR2GRAY), grayConversionThreshold, 255, 0)
            fingerContours, _ = cv2.findContours(circular_and_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            fingerContourCount = -1 # start at -1 to account for wrist
            for cntr in fingerContours:
                cntrArea = cv2.contourArea(cntr)
                if cntrArea > countedContourSize and cntrArea < circleCircumference * 0.25:
                    fingerContourCount += 1
            if fingerContourCount == -1: # if wrist too large, might fall below 0 -> shouldn't be below 0
                fingerContourCount = 0
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

        cv2.imshow("Video Feed", redAsWhiteFrame)
        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('q'):
            break
        if pressedKey == ord('f'):
            count += 1
            passwordAttempt.append((locFound, fingerContourCount))
            cv2.imwrite(dirName + "/cam-view-" + str(count) + ".jpg", frame)
            cv2.imwrite(dirName + "/mach-view-" + str(count) + ".jpg", redAsWhiteFrame)
        if pressedKey == ord('j'):
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
        # TODO
        # make all tests
        # make video test
        # clean up code
        # write documentation
