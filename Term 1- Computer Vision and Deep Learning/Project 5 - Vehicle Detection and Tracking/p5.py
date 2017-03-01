# Step 1:
# Make a list of images to read in

import os
import glob

basedir = 'vehicles/'
image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
    cars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Vehicle Images found:', len(cars))
with open("cars.txt", 'w') as f:
    for fn in cars:
        f.write(fn+'\n')

basedir = 'non-vehicles/'
image_types = os.listdir(basedir)
notcars = []
for imtype in image_types:
    notcars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Non-Vehicle Images found:', len(notcars))
with open("notcars.txt", 'w') as f:
    for fn in notcars:
        f.write(fn+'\n')

# Step 2:
# Get some sample car and not-car images and apply feature extraction
# Then visualize the original and feature extracted images

#%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from features import single_img_features, extract_features, apply_threshold
from window import visualize, slide_window, search_windows, draw_boxes, find_cars, draw_labeled_bboxes

# Choose a single sample of random car / not-car indices
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Define feature extraction parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# YCrCb tends to pickup gradients very well
# In this color space, the influence of illumination difference is reduced
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions - changing this to 32 reduces accuracy
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# Extract features from car / not-car images
car_features, car_hog_image = single_img_features(car_image, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

# Visualize the car / not-car images
images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
titles = ['car image', 'car HOG image', 'notcar image', 'notcar HOG image']
fig = plt.figure(figsize=(12,3))#, dpi=80)
visualize(fig, 1, 4, images, titles)

# Step 3:
# Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier.
# Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
# Normalize your features and randomize a selection for training and testing

# Redefine some feature extraction parameters
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

t=time.time()
# n_samples = 1000
# random_idxs = np.random.randint(0, len(cars), n_samples)
test_cars = cars #np.array(cars)[random_idxs]
test_notcars = notcars #np.array(notcars)[random_idxs]

# Extract features from car / not-car images
print('Computing features...')

car_features = extract_features(test_cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(test_notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

print(round(time.time()-t, 2), 'seconds to compute features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64) # StandardScaler expects np.float64
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state) #test_size=0.2 on lesson

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Wrap linear SVC on calibrated classifier so we can get probabilities
clf = CalibratedClassifierCV(svc)
# Check the training time for the SVC
t=time.time()
print('Training svc...')
clf.fit(X_train, y_train)
print(round(time.time()-t, 2), 'seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))

# Step 4:
# Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# Use the sample images from the video to test classifier and extract HOG features using multiple window sizes
searchpath = 'test_images/*'
example_images = glob.glob(searchpath)
images = []
titles = []
for img_src in example_images:
    t1 = time.time()
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    # we trained with .png images (scaled 0 to 1 by mpimp)
    # so we have to scale our test .jpg images (scaled 0 to 255) to the same scale
    img = img.astype(np.float32)/255

    # We use 4 window sizes, X-LARGE: 420x420, LARGE: 240x240, MEDIUM: 120x120 and SMALL: 60x60
    w1 = slide_window(img, x_start_stop=[500, 920], y_start_stop=[400, 460], xy_window=(60, 60), xy_overlap=(0.5, 0.5))
    w2 = slide_window(img, x_start_stop=[380, 1100], y_start_stop=[370, 490], xy_window=(120, 120), xy_overlap=(0.5, 0.5))
    w3 = slide_window(img, x_start_stop=[70, 1270], y_start_stop=[310, 550], xy_window=(240, 240), xy_overlap=(0.5, 0.5))
    w4 = slide_window(img, x_start_stop=[10, 1270], y_start_stop=[220, 640], xy_window=(420, 420), xy_overlap=(0.5, 0.5))
    windows = w1 + w2 + w3 + w4

    hot_windows = search_windows(img, windows, clf, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
    images.append(window_img)
    titles.append('')
    print(round(time.time()-t1, 2), 'seconds to process one image searching', len(windows), 'windows')

# visualize sliding-window results
fig = plt.figure(figsize=(16,6))
visualize(fig, 2, 3, images, titles)

# Extract HOG features just once for the entire region of interest in each full image / video frame using the find-cars helper function
# Try to eliminate multiple detections & false positives by using a heatmap and imposing a threshold
from scipy.ndimage.measurements import label
# We use the label function to find individual contiguous pixels in the heatmap

out_titles = []
out_images = []
out_maps = []
ystart = 400
ystop = 656
scale = 1.5
threshold = 1 # number of detections needs to be more than this to stay in the heatmap
# Iterate over test images
for img_src in example_images:
    img = mpimg.imread(img_src)
    # find cars
    out_img, heat_map = find_cars(img, scale, hog_channel, ystart, ystop, orient, pix_per_cell,
                                  cell_per_block, spatial_size, hist_bins, 
                                  clf, scl=X_scaler, prob_threshold=0.8, cps=1)
    # apply threshold
    heat_map = apply_threshold(heat_map, threshold)
    labels = label(heat_map)
    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    out_images.append(draw_img)
    out_images.append(heat_map)
    out_titles.append(img_src[-9:])
    out_titles.append(img_src[-9:])

# visualize full image HOG extractiong results
fig = plt.figure(figsize=(24,8))
visualize(fig, 3, 4, out_images, out_titles)

# Step 5:
# Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# Estimate a bounding box for vehicles detected.
def process_image(img):
    out_img, heat_map = find_cars(img, scale, hog_channel, ystart, ystop, orient, pix_per_cell,
                                  cell_per_block, spatial_size, hist_bins, 
                                  clf, scl=X_scaler, prob_threshold=0.8, cps=1)
    heat_map = apply_threshold(heat_map, threshold)
    labels = label(heat_map)
    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

test_output = 'test.mp4'
clip = VideoFileClip('project_video.mp4')
test_clip = clip.fl_image(process_image)
test_clip.write_videofile(test_output, audio=False)

HTML("""
<video width="960" height="540" controls>
    <source src="{0}">
</video>
""".format(test_output))