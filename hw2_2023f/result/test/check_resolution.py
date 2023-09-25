import cv2

# Load the two images
test_image = cv2.imread('/Users/treblocami/Desktop/job/cs484/hw2_2023f/data/cat.bmp', -1) / 255.0
test_image = cv2.resize(test_image, dsize=None, fx=0.7, fy=0.7, )
image2 = cv2.imread('/Users/treblocami/Desktop/job/cs484/hw2_2023f/result/test/sobel_image.jpg')

# Get the dimensions (resolution) of each image
height1, width1, channels1 = test_image.shape
height2, width2, channels2 = image2.shape

# Compare the resolutions
if (height1, width1) == (height2, width2):
    print("Both images have the same resolution:", (width1, height1))
else:
    print("Image 1 resolution:", (width1, height1))
    print("Image 2 resolution:", (width2, height2))
