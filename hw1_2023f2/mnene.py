import cv2
import numpy as np

# Load the image
I = cv2.imread('gigi.jpg').astype(np.uint8)

# Reduce brightness by subtracting 40, but clip values to stay in the 0-255 range
I = np.clip(I - 40, 0, 255)

# Save the result
cv2.imwrite('result.png', I)
