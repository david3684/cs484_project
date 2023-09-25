import time
import cv2
import numpy as np

start = time.time()

for i in range(1000):
    A = cv2.imread('grizzlypeakg.png',0)
    if A is None:
        print('Error')
    else:
        m1, n1 = A.shape
        for i in range(m1):
            for j in range(n1): 
                if A[i,j] <= 10:
                    A[i,j] = 0
end = time.time()
cv2.imwrite('result.png',A)
print(f"{end-start:.5f} sec")
