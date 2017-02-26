#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./pipeline_output_deploy/grayscale.png "Grayscale"
[image2]: ./pipeline_output_deploy/gaussian_blur.png "Gaussian Blur"
[image3]: ./pipeline_output_deploy/yellow_and_white.png "Yellow and White"
[image4]: ./pipeline_output_deploy/canny.png "Canny on the whole image"
[image5]: ./pipeline_output_deploy/img_roi.png "Image ROI (geometrical)"
[image6]: ./pipeline_output_deploy/edges_roi.png "Canny on ROI"
[image7]: ./pipeline_output_deploy/edges_roi_yellow.png "Canny on ROI and color mask"
[image8]: ./pipeline_output_deploy/lines_canny.png "Hough lines fitted on detected Canny"
[image9]: ./pipeline_output_deploy/hough_lines.png "Hough lines on image"
[image10]: ./pipeline_output_deploy/hough_lines_left_right.png "Houg lines sorted for left and right lane"
[image11]: ./pipeline_output_deploy/lane_poly_fit.png "Line polynomial fit - final result"

---

### Reflection

###1. Describe your pipeline.

My pipeline consisted a more steps, in order to accomodate the different
challenges on the road.

1. mask the image for yellow and white regions only (needed for the challenge video)
2. grayscale the masked input for canny edge detection
3. guassian blur for canny edge detection, to smooth out spurious edges
4. region of interest masking (geometrical masking of the region
 where the road is supposed to be located)
5. Canny edge detction for detecting the edges
6. Hough lines finding algorithm
7. Line filtering according to slope in left lane lines, and right lane lines
8. Fitting of a polynomial line for left and line segments
9. Line consolidation, and temporal filtering of fitted lines.


I modified the draw_lines() function to sort line segments in left and right lanes.
I used the slope information and considered to be a left line if slope < -0.5
and a right line if slope > 0.5.

Then, I've created a draw_lane() function to draw a specific fitted line,
given a set of x and y coordinates. The line is extrapolated using dynamic coordinates,
since the last challenge video has a different resolution. So, everything is drawn
in coordinates relative to the image like 0, int(imshape[1] / 2 - 5)

If lane finding pipeline, I've added the following:

1. mask_image() - method used for masking the image using a color range.
To narrow down the yellow and white regions, I had to convert the image to HSV format.
where the whites and yellows have a specific range.
2. mask_yellow_and_white() - method used for masking yellow and white regions
3. lane_detector(image) - main method for lane detection pipeline
I had to play around with hough transform parameters in order to filter out spurious matches.

In order to stabilize the detections, I've added temporal filtering.
The draw_lines() method updates 2 global variables, left_temp_lane_params, and right_temp_lane_params.
The draaw_lane() method uses temporal filtering by averaging the current line parameter
with the last frame parameter, using weighted average with alpha = 0.5.

lane_params = alpha * lane_params + (1-alpha) * last_params

On challenge video, due to extensive line filtering, there are some frames
where there are not enough x and y parameters for the right line.
So, when there is no detection, I simply take the last detected line (right/left)
in order to stabilize the detections.

In order to visualize the differnet stages of the pipeline,
I've created a directory, called pipeline_output, which show the same frame, at different stages
of detection:

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]



In video_output directory, I have included the lane detection result for
the 3 scenarios, white.mp4, yellow.mp4, extra.mp4

###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when more than 2 consecutive frames
generate false detetections or no detection at all.
The temporal filtering won't be able to compensate for the misdetections.

Also, the current pipeline has a very narrow applicable scenario:

1. only white yellow lines
2. straight lines
3. only 2 lines - left and right


###3. Suggest possible improvements to your pipeline

A possible improvement would be to generalize to any number of lanes, and lines
for any color and road shape. This would involve a more robust algorithm.
A deep learning segmentation algorithm might be required.
This approach still needs labeled segmented data.

For making my life easier, I've created separate files besides the notebook.
They are helpers.py, line_detector.py, and video_pipeline.py.

The code is also up to date in the jupiter notebook.