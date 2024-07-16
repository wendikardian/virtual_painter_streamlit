import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image
import os

# Define the hand detector class
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

# Load the header images
overlayList = []
header_path = "header"
if os.path.exists(header_path):
    myListDirectory = os.listdir(header_path)
    for imPath in myListDirectory:
        image = cv2.imread(f'{header_path}/{imPath}')
        overlayList.append(image)
else:
    st.error(f"The directory '{header_path}' does not exist. Please ensure it is present.")

# Initialize variables
if overlayList:
    header = overlayList[0]
else:
    header = np.zeros((125, 1280, 3), np.uint8)  # Create a black header if no images are found

drawColor = (0, 0, 255)
brushThickness = 7
eraserThickness = 40
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Streamlit app
st.title("Virtual Painter")
st.subheader("Draw with your fingers using a webcam")

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)
detector = handDetector(detectionConfidence=0.5)
cap.set(3, 1280)
cap.set(4, 720)

while run:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 320 < x1 < 480:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 480 < x1 < 630:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 650 < x1 < 840:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                elif x1 > 1000:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

        if fingers[1] and not fingers[2]:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    frameGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, frameInvers = cv2.threshold(frameGray, 50, 255, cv2.THRESH_BINARY_INV)
    frameInvers = cv2.cvtColor(frameInvers, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, frameInvers)
    frame = cv2.bitwise_or(frame, imgCanvas)
    frame[0:125, 0:1280] = header

    FRAME_WINDOW.image(frame)

cap.release()
