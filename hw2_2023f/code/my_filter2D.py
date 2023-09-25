import cv2
import numpy as np

def my_filter2D(image, kernel):
    # This function computes convolution given an image and kernel.
    # While "correlation" and "convolution" are both called filtering, here is a difference;
    # 2-D correlation is related to 2-D convolution by a 180 degree rotation of the filter matrix.
    #
    # Your function should meet the requirements laid out on the project webpage.
    #
    # Boundary handling can be tricky as the filter can't be centered on pixels at the image boundary without parts
    # of the filter being out of bounds. If we look at BorderTypes enumeration defined in cv2, we see that there are
    # several options to deal with boundaries such as cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, etc.:
    # https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    #
    # Your my_filter2D() computes convolution with the following behaviors:
    # - to pad the input image with zeros,
    # - and return a filtered image which matches the input image resolution.
    # - A better approach is to mirror or reflect the image content in the padding (borderType=cv2.BORDER_REFLECT_101).
    #
    # You may refer cv2.filter2D() as an exemplar behavior except that it computes "correlation" instead.
    # https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)   # for extra credit
    # Your final implementation should not contain cv2.filter2D().
    # Keep in mind that my_filter2D() is supposed to compute "convolution", not "correlation".
    #
    # Feel free to add your own parameters to this function to get extra credits written in the webpage:
    # - pad with reflected image content
    # - FFT-based convolution

    ################
    # Your code here
    ################
    channel = 1
    kernel_height, kernel_width = kernel.shape
    if kernel_height%2==0 or kernel_width%2 == 0:
        print('Error: The filter size is even\n')

    padding_width = kernel_width//2
    padding_height = kernel_height//2
    
    print(image.shape)
    if(image.ndim == 2): #grayscale image
        input_height, input_width = image.shape
        padded_image = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width, borderType=cv2.BORDER_REFLECT_101)
    else: #color image
        input_height, input_width, channel = image.shape
        padded_image = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width, borderType=cv2.BORDER_REFLECT_101)
    
    padded_kernel = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width, borderType=cv2.BORDER_REFLECT_101)
    output = np.zeros_like(image)
    

    print(padded_image.shape)
    
    kernelflipped = np.flipud(np.fliplr(kernel))
    for k in range(channel):
        for i in range(input_height):
            for j in range(input_width):
                patch = padded_image[i:i + kernel_height, j:j + kernel_width, k]
                output[i,j,k] = np.sum(patch*kernelflipped)
    print(image.size)
    print(output.size)
    return output
