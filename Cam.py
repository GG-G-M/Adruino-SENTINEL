import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# Simple test: open camera
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()
if ret:
    print("Camera is working")
cap.release()
