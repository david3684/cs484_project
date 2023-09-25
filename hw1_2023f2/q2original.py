import cv2
import numpy as np
I = cv2.imread('gigi.jpg').astype(np.uint8)
I = I - 40
cv2.imwrite('result.png', I)