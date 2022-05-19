import cv2
from objectDetector import detect

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    detect(frame, False)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
