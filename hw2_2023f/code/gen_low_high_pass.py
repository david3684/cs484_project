import cv2
import numpy as np
import time
import os
from my_filter2D import my_filter2D
from my_filter2D_fft import my_filter2D_fft


def hw2_testcase():
    # This script has test cases to help you test your my_filter2D() function. You should verify here that your
    # output is reasonable before using your my_filter2D to construct a hybrid image in hw2.py. The outputs are all
    # saved and you can include them in your writeup. You can add calls to cv2.filter2D() if you want to check that
    # my_filter2D() is doing something similar.
    #
    # Revised by Dahyun Kang and originally written by James Hays.

    ## Setup
    test_image = cv2.imread('/Users/treblocami/Desktop/job/cs484/hw2_2023f/data/cat.bmp', -1) / 255.0
    test_image = cv2.resize(test_image, dsize=None, fx=0.7, fy=0.7, )
    #image_height, image_width, channel = test_image.shape
    #print(image_height, image_width, channel)
    result_dir = '/Users/treblocami/Desktop/job/cs484/hw2_2023f/result/lowhighresult'
    os.makedirs(result_dir, exist_ok=True)

    cv2.imshow('test_image', test_image)
    cv2.imwrite(os.path.join(result_dir, 'input_image.jpg'), test_image * 255)
    cv2.waitKey(10)

    ##################################
    ## Identify filter
    # This filter should do nothing regardless of the padding method you use.
    low_pass_filter = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])

    low_pass_image = my_filter2D(test_image, low_pass_filter)

    cv2.imshow('identity_image', low_pass_image)
    cv2.imwrite(os.path.join(result_dir, 'low_pass_image.jpg'), low_pass_image * 255)

    high_pass_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    high_pass_image = my_filter2D(test_image, high_pass_filter)

    cv2.imshow('identity_image', high_pass_image)
    cv2.imwrite(os.path.join(result_dir, 'high_pass_image.jpg'), high_pass_image * 255)

    ##################################
    ## Done
    print('Press any key ...')
    cv2.waitKey(0)


if __name__ == '__main__':
    hw2_testcase()
