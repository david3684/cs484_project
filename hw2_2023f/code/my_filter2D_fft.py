import cv2
import numpy as np

def my_filter2D_fft(image, kernel):
    # Get the dimensions of the image and kernel
    padding=0
    image_height, image_width = image.shape[:2]

    # Calculate the dimensions of the padded image
    padded_height = image_height + 2 * padding
    padded_width = image_width + 2 * padding

    # Use np.pad() to apply zero-padding to the image
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Initialize an empty result array
    result = np.zeros_like(padded_image)

    # Loop through each color channel separately
    for channel in range(padded_image.shape[2]):
        # Extract the current color channel
        image_channel = padded_image[:, :, channel]

        # Perform FFT on the padded image channel and kernel
        image_fft = np.fft.fft2(image_channel)
        kernel_fft = np.fft.fft2(kernel, s=(padded_height, padded_width))

        # Compute the convolution in the frequency domain
        convolution_freq = image_fft * kernel_fft

        # Perform the inverse FFT to get the result in spatial domain
        result_channel = np.fft.ifft2(convolution_freq)

        # Take the real part of the result (discard the imaginary part)
        result_channel = np.real(result_channel)

        # Store the filtered channel in the result
        result[:, :, channel] = result_channel

    return result
