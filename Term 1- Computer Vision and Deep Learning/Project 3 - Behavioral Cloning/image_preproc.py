import cv2
import numpy as np

# Some of these image preprocessing functions are based on Vivek Yadav's augmentation code
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.nen5tsjgw

# Loads an image and changes the color space to RGB
def load_image(image_name):
    image_name = image_name.strip()
    #changing to RGB was crucial step in the image processing
    #as the simulator feeds RGB images, not BGR images
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    return image

# Halves the image size
def reduce_image(image):
    r_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    return r_image
    
# Randomly flips image horizontally and returns the mirrored steering angle
# this is done to reduce bias for a particular turning direction
def flip_image(image, angle):
    if np.random.randint(2) == 0:
        return cv2.flip(image, 1), -angle
    else:
        return image, angle
    
# Randomly translates image vertically and horizontally
# This is done to improve recovery and to simulate driving uphill/downhill
def trans_image(image, steer, t_range=100):
    rows,cols,ch = image.shape
    # Horizontal translation
    tr_x = t_range * np.random.uniform() - t_range / 2
    # New steering angle
    n_steer = steer + tr_x / t_range * 2 * 0.2
    # Vertical translation
    tr_y = 40 * np.random.uniform() - 40 / 2
    # Translation matrix to be used for affine transformation
    Trans_M = np.float32([[1, 0, tr_x],[0, 1, tr_y]])
    t_image = cv2.warpAffine(image, Trans_M, (cols,rows))
    return t_image, n_steer

# Crop top 68 pixels and bottom 20 pixels
# This is the equivalent of removing the sky and the car hood
def crop_image(image):
    shape = image.shape
    crop_image = image[68:shape[0]-20, 0:shape[1]]
    return crop_image
    
# Change image color space to HSV
# Randomly scale V channel to increase/reduce brightness
# Return image color space to RGB
# This helps with shadows and driving with other different light conditions
def scale_brightness_image(image):
    temp = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    b_scale = 0.25 + np.random.uniform() # range [0.25, 1.25)
    # We use Numpy indexing instead of cv2.split and cv2.merge
    # as those operations are more costly
    # [:, :, 2] is the V channel
    temp[:, :, 2] = temp[:, :, 2] * b_scale
    scaled_image = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)    
    return scaled_image