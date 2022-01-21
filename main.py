import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import math
import numpy as np
import random

# Distance function
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]
coff = np.polyfit(x, y, 2)

# Camera settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
handDetector = HandDetector(detectionCon=0.8, maxHands=1)

# Draw button
circleX, circleY = random.randint(150, 1100), random.randint(150, 500)
circlePos = random.randint(20, 90)
circleColor = (255, 0, 255)
counter = 0

pTime = 0

# Upload images
overlayMass = []
for i in range(1, 6):
    overlayMass.append(cv2.imread("pokemon" + str(i) + ".png", cv2.IMREAD_UNCHANGED))
overlayPick = overlayMass[random.randint(0, len(overlayMass) - 1)]
overlaySize = [150, 150]
success, img = cap.read()

clickBlock = True
score = 0

# Stream from camera
while True:
    # Read picture from cam
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = handDetector.findHands(img)

    # Add overlay
    overlay = cv2.resize(overlayPick, (overlaySize[0] - circlePos, overlaySize[1] - circlePos), None, 0.3, 0.3)
    img = cvzone.overlayPNG(img, overlay, [circleX, circleY])

    # Add text
    cv2.putText(img, f'{circlePos} cm', (circleX - 20, circleY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.putText(img, f'Score: {score} ', (1100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    if hands:
        lmList = hands[0]['lmList']
        bbox = hands[0]['bbox']
        x1, y1 = lmList[5]
        x2, y2 = lmList[17]

        # Get distance to hand
        distance = int(math.sqrt((y2-y1) ** 2 + (x2-x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C

        print(clickBlock)
        if distanceCM > circlePos:
            clickBlock = False


        if distanceCM < circlePos and not clickBlock:
            if bbox[0] < circleX + overlaySize[0] / 2 < bbox[0] + bbox[2] and \
                    bbox[1] < circleY + overlaySize[1] / 2 < bbox[1] + bbox[3]:
                counter = 1

        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (bbox[0], bbox[1]))

    if counter:
        counter += 1
        circleColor = (0, 255, 0)
        if counter >= 3:
            # Spawn new poke
            clickBlock = True
            overlayPick = overlayMass[random.randint(0, len(overlayMass) - 1)]
            circleX, circleY = random.randint(100, 1100), random.randint(100, 600)
            circlePos = random.randint(20, 90)
            circleColor = (255, 0, 255)
            counter = 0
            score += 1

    # Show FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show stream
    cv2.imshow('image', img)
    cv2.waitKey(1)


