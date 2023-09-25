import cv2
import numpy as np
I = cv2.imread('gigi.jpg').astype(np.uint8)
hsv_I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
mask1 = hsv_I[:,:,2] < 40
hsv_I[mask1] = 0

mask2 = hsv_I[:,:,2] != 0

hsv_I[mask2,2] = hsv_I[mask2,2] - 40
I = cv2.cvtColor(hsv_I, cv2.COLOR_HSV2BGR)
cv2.imwrite('result.png', I)