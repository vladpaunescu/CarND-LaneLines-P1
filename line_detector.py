# then save them to the test_images directory.

from helpers import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

test_dir = 'test_images'
# test_dir = 'challenge_frames'
files = os.listdir(test_dir)
output_dir = 'pipeline_output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def log_output(img, stage):
    result = os.path.join(output_dir, "{}.png".format(stage))
    mpimg.imsave(result, img)


def mask_image(image, lower_range, upper_range):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Threshold the HSV image to get only selected range colors
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    return mask


def mask_yellow_and_white(image):
    # define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    masked_yellow = mask_image(image, lower_yellow, upper_yellow)
    #     plt.figure(2)
    #     plt.imshow(masked_yellow)

    #     # define range of white color in HSV
    sensitivity = 30
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    masked_white = mask_image(image, lower_white, upper_white)

    mask = cv2.bitwise_or(masked_yellow, masked_white)
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    return masked_img

DEBUG = True
def lane_detector(image):
    # STAGE 1 filter white and yellow blobs
    masked_img = mask_yellow_and_white(image)
    # return masked_img
    log_output(masked_img, "yellow_and_white")

    # STAGE 2 gayscale the input
    gray = grayscale(masked_img)
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    log_output(gray_color, "grayscale")

    # STAGE 2 gaussian blur
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    blur_gray_color = cv2.cvtColor(blur_gray, cv2.COLOR_GRAY2RGB)
    log_output(blur_gray_color, "gaussian_blur")

    # STAGE 3 Canny edge detection
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    log_output(edges_color, "canny")

    # STAGE 3 ROI filtering
    # Next we'll create a masked edges image using cv2.fillPoly()
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),
                          (imshape[1] / 2, imshape[0] / 2 + 20), (imshape[1] / 2 + 10, imshape[0] / 2 + 20),
                          (imshape[1], imshape[0])]],
                        dtype=np.int32)
    #     vertices = np.array([[(0,imshape[0]),(480,310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    masked_edges_color = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2RGB)
    log_output(masked_edges_color, "edges_roi")


    # debug stuff
    if DEBUG:
        masked_img = region_of_interest(image, vertices)
        log_output(masked_img, "img_roi")

    # STAGE 4 HOUGH LINES
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the image
    line_detections = weighted_img(lines, image, α=0.8, β=1., λ=0.)

    if DEBUG:
        canny_lines = weighted_img(lines, masked_edges_color, α=0.8, β=1., λ=0.)
        log_output(canny_lines, "lines_canny")

    return line_detections


if __name__ == "__main__":
    for file in files[2:]:
        image = mpimg.imread(os.path.join(test_dir, file))
        line_detections = lane_detector(image)
        print(file)
        basename = os.path.basename(file)
        fname = os.path.splitext(basename)[0]
        result = os.path.join(output_dir, "lane_{}.png".format(fname))
        mpimg.imsave(result, line_detections)
        break
