import cv2
from objectDetector import detect

image = cv2.imread('test.jpg')
detect(image, True)