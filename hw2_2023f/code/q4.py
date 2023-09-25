import cv2
import time
import numpy as np
from my_filter2D import my_filter2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filter_sizes = list(range(3, 16, 2))
image_sizes = [0.25, 0.5, 1, 2, 4, 8]

computation_times = np.zeros((len(filter_sizes), len(image_sizes)))

test_image = cv2.imread('/Users/treblocami/Downloads/hw2_2023f/questions/RISDance.jpg')

for i, filter_size in enumerate(filter_sizes):
    for j, image_size in enumerate(image_sizes):
        target_pixels = image_size * 1e6  # 1 megapixel = 1,000,000 pixels
        aspect_ratio = float(test_image.shape[1]) / float(test_image.shape[0])
        image_width = int((target_pixels * aspect_ratio) ** 0.5)
        image_height = int(target_pixels / image_width)
        resized_image = cv2.resize(test_image, (image_width,image_width))

        kernel = np.random.randn(filter_size, filter_size)

        start_time = time.time()
        result = my_filter2D(resized_image, kernel)
        end_time = time.time()
        computation_times[i, j] = end_time - start_time

X, Y = np.meshgrid(image_sizes, filter_sizes)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, computation_times, cmap='viridis')

# Set labels
ax.set_xlabel('Image Size (MPix)')
ax.set_ylabel('Filter Size')
ax.set_zlabel('Computation Time (s)')
ax.set_title('Computation Time vs. Filter Size and Image Size')

# Show the plot
plt.show()