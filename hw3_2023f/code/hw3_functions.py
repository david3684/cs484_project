################################################################
# WARNING
# --------------------------------------------------------------
# When you submit your code, do NOT include blocking functions
# in this file, such as visualization functions (e.g., plt.show, cv2.imshow).
# You can use such visualization functions when you are working,
# but make sure it is commented or removed in your final submission.
#
# Before final submission, you can check your result by
# set "VISUALIZE = True" in "hw3_main.py" to check your results.
################################################################
from utils import normalize_points
import numpy as np
import cv2
import matplotlib.pyplot as plt


#=======================================================================================
# Your best hyperparameter findings here
WINDOW_SIZE = 7
DISPARITY_RANGE = 40
AGG_FILTER_SIZE = 5



#=======================================================================================
def bayer_to_rgb_bilinear(bayer_img):
    ################################################################
    height, width = bayer_img.shape
    rgb_img = np.zeros((height + 2, width + 2, 3), dtype=np.uint8)
    
# 패딩을 고려하여 값을 채웁니다. 패딩으로 인해 인덱스가 1씩 밀립니다.
    rgb_img[1:-1:2, 1:-1:2, 0] = bayer_img[0::2, 0::2]    
    rgb_img[1:-1:2, 2:-1:2, 1] = bayer_img[0::2, 1::2]
    rgb_img[2:-1:2, 1:-1:2, 1] = bayer_img[1::2, 0::2]
    rgb_img[2:-1:2, 2:-1:2, 2] = bayer_img[1::2, 1::2]
    
    
    
    # For the R channel
    print(rgb_img[0:10,0:10,2])
    rgb_img[1::2, 2:-3:2, 0] = (rgb_img[1::2, 1:-3:2,0]//2 + rgb_img[1::2, 3:-1:2,0]//2) 
    rgb_img[2:-3:2, 1::2, 0] = (rgb_img[1:-3:2, 1::2,0]//2 + rgb_img[3:-1:2, 1::2,0]//2) 
    
    
    rgb_img[2:-3:2, 2:-3:2, 0] = (
    rgb_img[1:-3:2, 1:-3:2, 0]//4 + rgb_img[1:-3:2, 3:-1:2, 0]//4 +
    rgb_img[3:-1:2, 1:-3:2, 0]//4 + rgb_img[3:-1:2, 3:-1:2, 0]//4
) 
    print(rgb_img[0:10,0:10,2])
    #cv2.imwrite('/Users/treblocami/Desktop/job/cs484_project/hw3_2023f/result/rgb_r.png', rgb_img[:,:,2])
    
    #np.savetxt('/Users/treblocami/Desktop/job/cs484_project/hw3_2023f/result/rgb_img.txt', rgb_img.reshape(-1, rgb_img.shape[-1]), fmt='%d', delimiter=', ')
    # For the G channel
    # For the G channel
    # Take care of array shapes during slicing
    # Calculate the size of the smallest possible aligned sub-arrays
    print(rgb_img[0:10,0:10,1])
    # 첫 번째 연산
    rgb_img[1:-1:2, 1:-1:2, 1] = (
        rgb_img[1:-1:2, 0:-2:2, 1]//4 + rgb_img[0:-2:2, 1:-1:2, 1]//4 +
        rgb_img[2::2, 1:-1:2, 1]//4 + rgb_img[1:-1:2, 2::2, 1]//4
    )
    

    # 두 번째 연산
    rgb_img[2:-2:2, 2:-2:2, 1] = (
        rgb_img[2:-2:2, 1:-3:2, 1]//4 + rgb_img[1:-3:2, 2:-2:2, 1]//4 +
        rgb_img[3:-1:2, 2:-2:2, 1]//4 + rgb_img[2:-2:2, 3:-1:2, 1]//4
    )
    print(rgb_img[0:10,0:10,1])
    
    #cv2.imwrite('/Users/treblocami/Desktop/job/cs484_project/hw3_2023f/result/rgb_g.png', rgb_img[:,:,1])

    
    print(rgb_img[0:10,0:10,0])
    # For the B channel
    rgb_img[2::2, 1:-2:2, 2] = (rgb_img[2::2, 0:-3:2,2]//2 + rgb_img[2::2, 2:-1:2,2]//2) 
    rgb_img[1:-2:2, 2::2, 2] = (rgb_img[0:-3:2, 2::2,2]//2 + rgb_img[2:-1:2, 2::2,2]//2) 
    rgb_img[1:-2:2, 1:-2:2, 2] = (
    rgb_img[0:-3:2, 0:-3:2, 2]//4 + rgb_img[0:-3:2, 2:-1:2, 2]//4 +
    rgb_img[2:-1:2, 0:-3:2, 2]//4 + rgb_img[2:-1:2, 2:-1:2, 2]//4
)
    print(rgb_img[0:10,0:10,0])
    #print(rgb_img[0:10,0:10,0])
    # 마지막에 패딩 제거
    rgb_img = rgb_img[1:-1, 1:-1, :]

    #cv2.imwrite('/Users/treblocami/Desktop/job/cs484_project/hw3_2023f/result/rgb_b.png', rgb_img[:,:,0])
    ################################################################
    return rgb_img



#=======================================================================================
def bayer_to_rgb_bicubic(bayer_img):
    # Your code here
    ################################################################
    rgb_img = None


    ################################################################
    return rgb_img



#=======================================================================================
def calculate_fundamental_matrix(pts1, pts2):
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2
    assert pts1.shape[0] == pts2.shape[0]
    
    n = pts1.shape[0]
    
    # Normalization
    mean1 = np.mean(pts1, axis=0)
    mean2 = np.mean(pts2, axis=0)
    
    scale1 = np.sqrt(2) / np.std(pts1)
    scale2 = np.sqrt(2) / np.std(pts2)
    
    T1 = np.array([
        [scale1, 0, -scale1 * mean1[0]],
        [0, scale1, -scale1 * mean1[1]],
        [0, 0, 1]
    ])
    
    T2 = np.array([
        [scale2, 0, -scale2 * mean2[0]],
        [0, scale2, -scale2 * mean2[1]],
        [0, 0, 1]
    ])
    
    pts1 = np.column_stack([pts1, np.ones(n)])
    pts2 = np.column_stack([pts2, np.ones(n)])

    pts1_normalized = (T1 @ pts1.T).T
    pts2_normalized = (T2 @ pts2.T).T
    
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1, _ = pts1_normalized[i]
        x2, y2, _ = pts2_normalized[i]
        A[i] = [x1*x2, x2*y1, x2, y2*x1, y1*y2, y2, x1, y1, 1]
    
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1]
    F = f.reshape(3, 3)
    
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt
    
    # Denormalize
    F = T2.T @ F @ T1
    
    return F



#=======================================================================================
def transform_fundamental_matrix(F, h1, h2):
    # Your code here
    ################################################################
    F_mod = np.linalg.inv(h2).T @ F @ np.linalg.inv(h1)
    
    
    ################################################################
    return F_mod



#=======================================================================================
def rectify_stereo_images(img1, img2, h1, h2):
    # Your code here
    # You should get un-cropped image.
    # In order to superpose two rectified images, you need to create certain amount of margin.
    # Which means you need to do some additional things to get fully warped image (not cropped).
    ################################################
    height, width = img1.shape[:2]
    
    # Create translation matrices
    T1 = np.array([[1, 0, width // 4],
                   [0, 1, height // 4],
                   [0, 0, 1]])
    
    T2 = np.array([[1, 0, width // 4],
                   [0, 1, height // 4],
                   [0, 0, 1]])
    
    # Create scaling matrices
    S1 = np.array([[0.75, 0, 0],
                   [0, 0.75, 0],
                   [0, 0, 1]])
    
    S2 = np.array([[0.75, 0, 0],
                   [0, 0.75, 0],
                   [0, 0, 1]])
    
    # Modify homography matrices
    h1_mod = T1 @ S1 @ h1
    h2_mod = T2 @ S2 @ h2
    
    # Calculate new size for rectification
    rectified_size = (int(width * 1.5), int(height * 1.5))

    # Warp the images
    img1_rectified = cv2.warpPerspective(img1, h1_mod, rectified_size)
    img2_rectified = cv2.warpPerspective(img2, h2_mod, rectified_size)
    
    return img1_rectified, img2_rectified, h1_mod, h2_mod




#=======================================================================================
def calculate_disparity_map(img1, img2):
    
    # First convert color image to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # You have to get disparity (depth) of img1 (left)
    # i.e., I1(u) = I2(u + d(u)),
    # where u is pixel positions (x,y) in each images and d is dispairty map.
    # Your code here
    window_size=20
    d_max=50
    
    
    h, w = img1_gray.shape
    half_window = window_size // 2
    
    # Initialize disparity map and cost volume
    disparity_map = np.zeros((h, w), dtype=np.float32)
    cost_volume = np.zeros((h, w, d_max), dtype=np.float32)  

    cost_volume = np.zeros((h, w,d_max), dtype=np.float32)

    for d in range(d_max):
        print(f"Calculating disparity {d}")

        # Efficiently roll the image
        img2_shifted = np.roll(img2_gray, d, axis=1)

        for y in range(half_window, h - half_window):
            # Vectorized inner loop
            w1 = img1_gray[y - half_window:y + half_window + 1, half_window:w - half_window]
            w2 = img2_shifted[y - half_window:y + half_window + 1, half_window:w - half_window]
            
            mean_w1 = np.mean(w1, axis=0)
            mean_w2 = np.mean(w2, axis=0)
            
            numerator = np.sum((w1 - mean_w1) * (w2 - mean_w2), axis=0)
            denominator = np.sqrt(np.sum((w1 - mean_w1)**2, axis=0) * np.sum((w2 - mean_w2)**2, axis=0))

            ncc = np.where(denominator == 0, -1, numerator / denominator)
            cost_volume[y, half_window:w - half_window, d] = -ncc

    # Aggregate costs
    guide_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Assuming img1 is the reference image
    radius = 40
    epsilon = 0.2**2  # Epsilon in the Guided Filter
    gf = cv2.ximgproc.createGuidedFilter(guide_img, radius, epsilon)

    # Apply the Guided Filter to each disparity slice of the cost volume
    for d in range(d_max):
        cost_volume[:, :, d] = gf.filter(cost_volume[:, :, d])

    # Select disparity with minimum cost
    
    # Get disparity map
    disparity_map = -np.argmin(cost_volume, axis=2)
    
    epsilon = 1e-5  # Smoothing term to prevent division by zero
    max_subpixel_correction = 0.5

    for y in range(h):
        for x in range(w):
            d = disparity_map[y, x]
            if d == 0 or d == d_max - 1:
                continue
            C0 = cost_volume[y, x, d-1]
            C1 = cost_volume[y, x, d]
            C2 = cost_volume[y, x, d+1]

            denominator = 2*(C2 + C0 - 2*C1)
            
            # Check if the denominator is too small or zero
            if np.abs(denominator) < epsilon:
                continue

            # Compute subpixel correction
            subpixel_correction = (C2 - C0) / denominator

            # Limit the range of subpixel correction
            subpixel_correction = np.clip(subpixel_correction, -max_subpixel_correction, max_subpixel_correction)

            disparity_map[y, x] = d + subpixel_correction
    return disparity_map



#=======================================================================================
# Anything else:
