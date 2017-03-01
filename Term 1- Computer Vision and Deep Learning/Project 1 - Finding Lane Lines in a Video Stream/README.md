
# **Finding Lane Lines on the Road**
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below.

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note** If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="README_images/line-segments-example.jpg" width="380" alt="Combined Image"/>
 <figcaption>
 <p></p>
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p>
 </figcaption>
</figure>
 <p></p>
<figure>
 <img src="README_images/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p>
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p>
 </figcaption>
</figure>


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
```

    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x23a807c3eb8>




![png](README_images/output_3_2.png)


**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # Setup variables
    from math import floor
    left_line_count = 0
    right_line_count = 0
    left_average_slope = 0
    right_average_slope = 0
    left_average_x = 0
    left_average_y = 0
    right_average_x = 0
    right_average_y = 0
    left_min_y = left_max_y = left_min_y = right_min_y = right_max_y = img.shape[1]

    for line in lines:
        for x1,y1,x2,y2 in line:
            # Calculate each line slope
            slope = (y2-y1)/(x2-x1)
            if (not math.isinf(slope) and slope != 0):
                # Classify lines by slope (lines at the right have positive slope)
                # Calculate total slope, x and y to then find average
                # Calculate min y
                if (slope >= 0.5 and slope <= 0.85):
                    right_line_count += 1
                    right_average_slope += slope
                    right_average_x += x1 + x2
                    right_average_y += y1 + y2
                    if right_min_y > y1: right_min_y = y1
                    if right_min_y > y2: right_min_y = y2
                elif (slope <= -0.5 and slope >= -0.85):
                    left_line_count += 1
                    left_average_slope += slope
                    left_average_x += x1 + x2
                    left_average_y += y1 + y2
                    if left_min_y > y1: left_min_y = y1
                    if left_min_y > y2: left_min_y = y2
    if ((left_line_count != 0) and (right_line_count != 0)):
        # Find average slope for each side
        left_average_slope = left_average_slope / left_line_count
        right_average_slope = right_average_slope / right_line_count
        # Find average x and y for each side
        left_average_x = left_average_x / (left_line_count * 2)
        left_average_y = left_average_y / (left_line_count * 2)
        right_average_x = right_average_x / (right_line_count * 2)
        right_average_y = right_average_y / (right_line_count * 2)
        # Find y intercept for each side
        # b = y - mx
        left_y_intercept = left_average_y - left_average_slope * left_average_x
        right_y_intercept = right_average_y - right_average_slope * right_average_x
        # Find max x values for each side
        # x = ( y - b ) / m
        left_max_x =  floor((left_max_y - left_y_intercept) / left_average_slope)
        right_max_x = floor((right_max_y - right_y_intercept) / right_average_slope)
        # Find min x values for each side
        left_min_x =  floor((left_min_y - left_y_intercept) / left_average_slope)
        right_min_x = floor((right_min_y - right_y_intercept) / right_average_slope)
        # Draw left line
        cv2.line(img, (left_min_x, left_min_y), (left_max_x, left_max_y), color, thickness)
        # Draw right line
        cv2.line(img, (right_min_x, right_min_y), (right_max_x, right_max_y), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
```

## Test on Images

Now you should build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
# Remove any previously processed images
filelist = [ f for f in os.listdir('test_images/') if f.find('processed') != -1]
for f in filelist:
    print('Removing image:', 'test_images/' + f)
    os.remove('test_images/' + f)
test_images = os.listdir('test_images/')
for fname in test_images:
    # Get image path and name details
    basedir, basename = os.path.split(fname)
    root, ext = os.path.splitext(basename)
    # Read in an image
    image = mpimg.imread('test_images/' + basename)
    imshape = image.shape
    # Print out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape, 'and name', basename)
    # Make a grayscale copy of the image for processing
    gray = grayscale(image)
    # Define kernel size for Gaussian smoothing / blurring
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    # Define Canny transform paramerets
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    vertices = np.array([[(100,imshape[0]),(450, 325), (550, 325), (imshape[1]-100,imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 45     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20 # minimum number of pixels making up a line
    max_line_gap = 60    # maximum gap in pixels between connectable line segments
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    lines_edges = weighted_img(lines, image, α=0.8, β=1., λ=0.)
    print('Saving image:', root + '_processed.jpg')
    mpimg.imsave('test_images/' + root + '_processed.jpg', lines_edges)
```

    Removing image: test_images/solidWhiteCurve_processed.jpg
    Removing image: test_images/solidWhiteRight_processed.jpg
    Removing image: test_images/solidYellowCurve2_processed.jpg
    Removing image: test_images/solidYellowCurve_processed.jpg
    Removing image: test_images/solidYellowLeft_processed.jpg
    Removing image: test_images/whiteCarLaneSwitch_processed.jpg
    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3) and name solidWhiteCurve.jpg
    Saving image: solidWhiteCurve_processed.jpg
    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3) and name solidWhiteRight.jpg
    Saving image: solidWhiteRight_processed.jpg
    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3) and name solidYellowCurve.jpg
    Saving image: solidYellowCurve_processed.jpg
    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3) and name solidYellowCurve2.jpg
    Saving image: solidYellowCurve2_processed.jpg
    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3) and name solidYellowLeft.jpg
    Saving image: solidYellowLeft_processed.jpg
    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3) and name whiteCarLaneSwitch.jpg
    Saving image: whiteCarLaneSwitch_processed.jpg


run your solution on all test_images and make copies into the test_images directory).

## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    imshape = image.shape
    gray = grayscale(image)
    # Define kernel size for Gaussian smoothing / blurring
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    # Define Canny transform paramerets
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    vertices = np.array([[(100,imshape[0]),(450, 325), (550, 325), (imshape[1]-100,imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 45     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20 # minimum number of pixels making up a line
    max_line_gap = 60    # maximum gap in pixels between connectable line segments
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    result = weighted_img(lines, image, α=0.8, β=1., λ=0.)
    return result
```

Let's try the one with the solid white lane on the right first ...


```python
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video white.mp4
    [MoviePy] Writing video white.mp4


    100%|███████████████████████████████████████████████████████████████████████████████▋| 221/222 [00:02<00:00, 91.19it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: white.mp4

    Wall time: 2.71 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="white.mp4">
</video>




**At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video yellow.mp4
    [MoviePy] Writing video yellow.mp4


    100%|███████████████████████████████████████████████████████████████████████████████▉| 681/682 [00:07<00:00, 93.87it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: yellow.mp4

    Wall time: 7.52 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="yellow.mp4">
</video>




## Reflections

Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?

Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!

The current algorithm is likely to fail with:
1. Curved lane lines
2. Different lighting conditions on the road
3. Vertical lane lines (infinite slope)
4. Lane lines that slope in the same direction

I can imagine making my algorithm better or more robust by
1. Instead of interpolating into a line, interpolate into a curve, maybe a bezier with several control points.
2. Analyze the contrast/brightness in the area of interest and even (average?) it out so darker areas become lighter.
3. Treat vertical lines as a separate scenario and either ignore them or assign some default values
4. Separate the left and the right side of the image and analyze the lines on each side independently

## Submission

If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
```

    [MoviePy] >>>> Building video extra.mp4
    [MoviePy] Writing video extra.mp4


    100%|████████████████████████████████████████████████████████████████████████████████| 251/251 [00:06<00:00, 40.15it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: extra.mp4

    Wall time: 6.79 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```





<video width="960" height="540" controls>
  <source src="extra.mp4">
</video>





```python

```
