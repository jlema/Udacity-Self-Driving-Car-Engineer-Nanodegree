import numpy as np
import cv2
import glob
from scipy.misc import imread

rows, cols = 720, 1280 # camera format
# source and destination points are defined from top left then going clockwise
# define 4 source points
c_src = np.float32([(555, 475), (730, 475), (1095, 705), (215, 705)])
# define 4 destination points
offset = 250 # x-offset
c_dst = np.float32([[offset, 0], [cols-offset, 0],
                  [cols-offset, rows],
                  [offset, rows]])

# returns ret, mtx, dist, rvecs, tvecs
def cal_camera(cal_path='camera_cal/', fname_pattern='calibration*.jpg', debug=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob(cal_path + fname_pattern)
    
    # Step through the list and search for chessboard corners
    print('Starting camera calibration!')
    for fname in images:
        img = imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            if (debug):
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (cols, rows), None, None)
    
    # Do camera calibration given object points and image points
    print('Finished camera calibration!')
    return ret, mtx, dist, rvecs, tvecs 
    

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped