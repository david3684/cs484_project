import cv2
import time
import numpy as np


start = time.time()
for i in range(1000):
    A = cv2.imread('grizzlypeakg.png',0)
    if A is None:
        print('Error')
    B = A <= 10
    A[B] = 0
end = time.time()
#cv2.imwrite('result.png',A)
print(f"{end-start:.5f} sec")