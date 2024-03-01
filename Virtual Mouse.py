import cv2
import numpy as np
import time
import HandTracking as ht
import autopy   

pTime = 0               # Used to calculate frame rate
width = 1280            # Width of Camera
height = 720            # Height of Camera
frameR = 100            # Frame Rate
smoothening = 8         # Smoothening Factor
prev_x, prev_y = 0, 0   # Previous coordinates
curr_x, curr_y = 0, 0   # Current coordinates
dragging = False        # Flag to indicate if dragging is in progress
start_x, start_y = 0, 0 # Starting coordinates of the drag

cap = cv2.VideoCapture(0)   # Getting video feed from the webcam
cap.set(3, width)           # Adjusting size
cap.set(4, height)

detector = ht.handDetector(maxHands=1)                  
screen_width, screen_height = autopy.screen.size()      # Getting the screen size

while True:
    success, img = cap.read()
    img = detector.findHands(img)                       # Finding the hand
    lmlist, bbox = detector.findPosition(img)           # Getting position of hand

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # Index finger tip
        x2, y2 = lmlist[12][1:]  # Middle finger tip
        x3, y3 = lmlist[4][1:]  # Thumb tip
        x4, y4 = lmlist[20][1:]  # Tip of little finger
        x5, y5 = lmlist[16][1:]  # Tip of ring finger

        fingers = detector.fingersUp()      # Checking if fingers are upwards

        # Mouse movement
        if fingers[1] == 1 and fingers[2] == 0:     # If fore finger is up and middle finger is down (Move Mouse)
            x3 = np.interp(x1, (frameR, width-frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, height-frameR), (0, screen_height))

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            autopy.mouse.move(screen_width - curr_x, curr_y)    # Moving the cursor
            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y

        # Single click (Left click)
        if fingers[1] == 1 and fingers[2] == 1:     # If fore finger & middle finger both are up
            autopy.mouse.click(autopy.mouse.Button.LEFT)  # Perform Left Click

        # Right click
        if fingers[1] == 1 and fingers[0] == 1 and fingers[2] == 1:     # Fore finger, thumb, and middle finger are up
            autopy.mouse.click(autopy.mouse.Button.RIGHT)  # Perform Right Click

        # Dragging
        if fingers[0] == 1 and fingers[1] == 1:     # If thumb and index finger both are up (Drag with Pinch)
            length, img, lineInfo = detector.findDistance(8, 4, img)  # Distance between index and thumb tips
            if length < 40:     # If both fingers are really close to each other
                dragging = True
                start_x, start_y = curr_x, curr_y

        if dragging:  # If dragging is in progress
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, down=True)  # Hold left click for dragging

        if not all(fingers):  # If no fingers are up (End Drag)
            dragging = False
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, down=False)  # Release left click if dragged

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
