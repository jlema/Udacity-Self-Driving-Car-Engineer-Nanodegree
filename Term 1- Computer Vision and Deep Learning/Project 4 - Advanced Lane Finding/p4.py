'''
Advanced Lane Finding Project
=============================
The goals / steps of this project are the following:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

import cv2
import glob
import os
#%matplotlib qt

from matplotlib import pyplot as plt
from scipy.misc import imread, imsave
from threshold import threshold_image
from camera import cal_camera, warper, rows, cols, c_src, c_dst, offset
from lanes import overlay_lanes

# 1. Compute the camera calibration using chessboard images
ret, mtx, dist, rvecs, tvecs = cal_camera()

# 2. Apply a distortion correction to raw images
#%matplotlib inline
# Remove any previously undistorted images
filelist = [ f for f in os.listdir('output_images/') if f.find('_undist') != -1]
for f in filelist:
    os.remove('output_images/' + f)

# Make a list of raw images
images = glob.glob('test_images/*.jpg')

# Step through the list and undistort images
for fname in images:
    # Read raw image
    img = imread(fname)

    # Get image path and name details
    basedir, basename = os.path.split(fname)
    root, ext = os.path.splitext(basename)

    # Undistort image
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Save undistorted image
    imsave('output_images/' + root + '_undist.jpg', dst)

# Display an example distorted and an undistorted image
img = imread('test_images/test5.jpg')
dst = imread('output_images/test5_undist.jpg')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)

# 3. Use color transforms, gradients, etc., to create a thresholded binary image.

# Remove any previously thresholded images
filelist = [ f for f in os.listdir('output_images/') if f.find('_thresh') != -1]
for f in filelist:
    os.remove('output_images/' + f)

# Make a list of raw images
images = glob.glob('output_images/*_undist.jpg')

# Step through the list and apply thresholds
for fname in images:
    # Read raw image
    img = imread(fname)

    # Get image path and name details
    basedir, basename = os.path.split(fname)
    root, ext = os.path.splitext(basename)
    root = root[:root.rfind('_')]

    # Threshold image
    combined = threshold_image(img)

    # Save thresholded image
    imsave('output_images/' + root + '_thresh.jpg', combined)

# Display an example undistorted and a thresholded  image
img = imread('output_images/test5_undist.jpg')
dst = imread('output_images/test5_thresh.jpg')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(dst, cmap='gray')
ax2.set_title('Combined Thresholds', fontsize=30)

# 4. Apply a perspective transform to rectify binary image ("birds-eye view").

# Remove any previously perspective transformed images
filelist = [ f for f in os.listdir('output_images/') if f.find('_trans') != -1]
for f in filelist:
    os.remove('output_images/' + f)

# Make a list of thresholded images
images = glob.glob('output_images/*_thresh.jpg')

# Step through the list and apply perspective transform
for fname in images:
    # Read raw image
    img = imread(fname)

    # Get image path and name details
    basedir, basename = os.path.split(fname)
    root, ext = os.path.splitext(basename)
    root = root[:root.rfind('_')]

    # Apply perspective transform
    warped = warper(img, c_src, c_dst)

    # Save perspective transformed image
    imsave('output_images/' + root + '_trans.jpg', warped)

## Display an example undistorted and a thresholded  image
dst = imread('output_images/test5_trans.jpg')
image = imread('output_images/test5_undist.jpg')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img, cmap='gray')
ax1.set_title('Combined Thresholds', fontsize=30)
ax2.imshow(dst, cmap='gray')
ax2.set_title('Perspective Transformed', fontsize=30)
# Plot source points
ax1.plot([555, 730], [475, 475], color='r', linestyle='-', linewidth=2)
ax1.plot([730, 1095], [475, 705], color='r', linestyle='-', linewidth=2)
ax1.plot([1095, 215], [705, 705], color='r', linestyle='-', linewidth=2)
ax1.plot([215, 555], [705, 475], color='r', linestyle='-', linewidth=2)
# Plot destination points
ax2.plot([offset, cols-offset], [0, 0], color='r', linestyle='-', linewidth=2)
ax2.plot([cols-offset, cols-offset], [0, rows], color='r', linestyle='-', linewidth=2)
ax2.plot([cols-offset, offset], [rows, rows], color='r', linestyle='-', linewidth=2)
ax2.plot([offset, offset], [rows, 0], color='r', linestyle='-', linewidth=2)

# 5. Detect lane pixels and fit to find the lane boundary.
   
# Remove any previously lane boundary overplotted images
filelist = [ f for f in os.listdir('output_images/') if f.find('_final') != -1]
for f in filelist:
    os.remove('output_images/' + f)

# Make a list of perspective transformed images
t_images = glob.glob('output_images/*_trans.jpg')
# Make a list of undistorted images
u_images = glob.glob('output_images/*_undist.jpg')

# Step through the list and find lane pixels
for tname, fname in zip(t_images, u_images):
    # Read raw images
    t_img = imread(tname)
    u_img = imread(fname)
    # Apply grayscale thresholding to convert perspective transformed images to BW with 1 and 0 values only
    (thresh, t_img) = cv2.threshold(t_img, 128, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Get image path and name details
    basedir, basename = os.path.split(tname)
    root, ext = os.path.splitext(basename)
    root = root[:root.rfind('_')]

    # Find lane pixels and overlay lane boundary and related info
    final = overlay_lanes(t_img, u_img)

    # Save perspective transformed image
    imsave('output_images/' + root + '_final.jpg', final)

# Display an example perspective transformed and a lane boundary overplotted  image
img = imread('output_images/test5_trans.jpg')
u_img = imread('output_images/test5_undist.jpg')
dst = imread('output_images/test5_final.jpg')

# Display pixels and polynomial fit
overlay_lanes(img, u_img, True)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img, cmap='gray')
ax1.set_title('Perspective Transformed', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Final Overplotted Image', fontsize=30)

# TODO:
# 1. Modularize everything (functions) - DONE
# 2. Add vehicle position text - DONE
# 3. Hook up video pipeline - DONE
# 4. Use class to track loss lines and recover
# 5. Bonus: add coloring to lanes

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Define image processing pipeline
def process_image(image):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    thresh = threshold_image(undist)
    trans = warper(thresh, c_src, c_dst)
    result = overlay_lanes(trans, undist)
    return result

# Hook up video to image processing pipeline
video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(video_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))