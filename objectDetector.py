
import cv2


fire_cascade = cv2.CascadeClassifier('haar_cascades/fire_cascade.xml')
birds_cascade = cv2.CascadeClassifier('haar_cascades/bird_cascade.xml')
cars_cascade = cv2.CascadeClassifier('haar_cascades/cars_cascade.xml')
fullbody_cascade = cv2.CascadeClassifier('haar_cascades/fullbody_cascade.xml')
banana_cascade = cv2.CascadeClassifier('haar_cascades/banana_cascade.xml')


def detect(frame, log):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5)
    birds = birds_cascade.detectMultiScale(frame, 1.2, 5)
    cars = cars_cascade.detectMultiScale(frame, 1.2, 5)
    fullbody = fullbody_cascade.detectMultiScale(frame, 1.2, 5)
    banana = banana_cascade.detectMultiScale(frame, 1.2, 5)

    if (log):
        print('fires detected:', len(fire))
        print('birds detected:', len(birds))
        print('cars detected:', len(cars))
        print('fullbody detected:', len(fullbody))
        print('banana detected:', len(banana))

    for (x,y,w,h) in fire:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        print("fire is detected")

    for (w,y,w,h) in birds:
        cv2.rectangle(frame,(w-20,y-20),(w+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, w:w+w]
        roi_color = frame[y:y+h, w:w+w]
        print("bird is detected")
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        print("car is detected")

    for (x,y,w,h) in fullbody:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        print("fullbody is detected")

    for (x,y,w,h) in banana:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        print("banana is detected")
