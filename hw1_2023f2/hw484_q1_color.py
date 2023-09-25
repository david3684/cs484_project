import cv2
import time
import numpy as np

A = cv2.imread('grizzlypeak.jpg')
hsv_A = cv2.cvtColor(A, cv2.COLOR_BGR2HSV) #change bgr to hsv

start = time.time()
B = hsv_A[:,:,2] <= 10
hsv_A[B] = 0
A = cv2.cvtColor(hsv_A, cv2.COLOR_HSV2BGR)
end = time.time()
print(f"{end-start:.5f} sec")