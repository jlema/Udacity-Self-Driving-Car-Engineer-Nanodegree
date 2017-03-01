import numpy as np
import cv2
from matplotlib import pyplot as plt
from camera import c_src, c_dst

def overlay_lanes(t_img, u_img, debug=False):
    
    # Get histogram
    histogram = np.sum(t_img[int(t_img.shape[0]/2):,:], axis=0)    

    # Find peaks to start lane detection
    # Split histogram into two halves horizontally
    mid = int(histogram.shape[0]/2)
    # Find left and right peaks
    l_peak = (np.argmax(histogram[:mid]), t_img.shape[0])
    r_peak = (np.argmax(histogram[mid:])+mid, t_img.shape[0])
    
    # define expected distance between lane pixels centers
    p_dist = 750
    # define search window width
    w_width = 100
    # define number of horizontal bands to cut image into
    bands = 10
    # define band height
    w_height = t_img.shape[0] / bands
    # Using the left and right peaks as starting point:
    # move upwards band height pixels
    # get all '1' pixels inside the window
    # calculate the centroid of the window
    # move upwards band height pixels centered at x on the centroid
    # repeat until we reach the top of the image
    c_height = 0
    l_pixels_x = l_peak[0] # left lane pixels x coordinates
    l_pixels_y = l_peak[1] # left lane pixels y coordinates
    r_pixels_x = r_peak[0] # right lane pixels x coordinates
    r_pixels_y = l_peak[1] # right lane pixels y coordinates
    
    for b in range(bands):
        c_height = int(t_img.shape[0] - (b * w_height))
        # calculate global left and right windows y-origin
        y_orig = int(c_height - w_height)
        # calculate global left and right windows x-origins
        l_x_orig = int(l_peak[0] - w_width/2)
        r_x_orig = int(r_peak[0] - w_width/2)
        l_window = t_img[y_orig:c_height,l_x_orig:l_x_orig+w_width]
        r_window = t_img[y_orig:c_height,r_x_orig:r_x_orig+w_width]
        # np.nonzero returns [rows, cols] of non-zero pixels or [y, x]
        new_l_pixels = np.nonzero(l_window)
        new_r_pixels = np.nonzero(r_window)
    
        # if we find pixels on the left lane, store them
        if (len(new_l_pixels[1])):
            # transform to global coordinates
            new_l_pixels_x = new_l_pixels[1] + l_x_orig
            l_pixels_x = np.append(l_pixels_x, new_l_pixels_x)
            l_pixels_y = np.append(l_pixels_y, new_l_pixels[0] + y_orig)
            # new peak is mean of column values
            l_peak = (int(np.mean(new_l_pixels_x)), y_orig)
        # otherwise skip to the next window
        else:
            l_peak = (l_peak[0], y_orig)
        # if we find pixels on the right lane, store them
        if (len(new_r_pixels[1])):
            # transform to global coordinates
            new_r_pixels_x = new_r_pixels[1] + r_x_orig
            r_pixels_x = np.append(r_pixels_x, new_r_pixels_x)
            r_pixels_y = np.append(r_pixels_y, new_r_pixels[0] + y_orig)
            # new peak is mean of column values
            r_peak = (int(np.mean(new_r_pixels_x)), y_orig)
        # otherwise skip to the next window
        else:
            r_peak = (r_peak[0], y_orig)
    
    # Add top pixels for left and right lanes (useful for lanes that cut off at the top)
    l_pixels_x = np.append(l_pixels_x, l_peak[0])
    l_pixels_y = np.append(l_pixels_y, 0)
    r_pixels_x = np.append(r_pixels_x, r_peak[0])
    r_pixels_y = np.append(r_pixels_y, 0)
    
    # 6. Determine the curvature of the lane and vehicle position with respect to center.
    # Fit a second order polynomial to each fake lane line   
    leftx = l_pixels_x
    lefty = l_pixels_y
    rightx = r_pixels_x
    righty = r_pixels_y
    
    if (leftx.size > 1 and rightx.size > 1):
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
        
        # Plot up the data
        if (debug):            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            ax1.set_title('Detected Pixels', fontsize=30)
            ax1.plot(leftx, lefty, 'o', color='red')
            ax1.plot(rightx, righty, 'o', color='blue')
            ax2.set_title('Polynomial Fit', fontsize=30)
            ax2.plot(left_fitx, lefty, color='green', linewidth=3)
            ax2.plot(right_fitx, righty, color='green', linewidth=3)
            plt.gca().invert_yaxis() # to visualize as we do the images
              
        # 7. Warp the detected lane boundaries back onto the original image.
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(t_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, righty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Calculate inverse perspective matrix
        Minv = cv2.getPerspectiveTransform(c_dst, c_src)
               
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (u_img.shape[1], u_img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(u_img, 1, newwarp, 0.3, 0)
        
        # 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
        # Calculate radius of curvature       
        # Define conversions in x and y from pixels space to meters
        l_y_eval = np.max(lefty)
        r_y_eval = np.max(righty)
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*l_y_eval + left_fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*r_y_eval + right_fit_cr[1])**2)**1.5) \
                                        /np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        # Example values: 3380.7 m    3189.3 m
        # Output curvature
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_roc = "Left RoC: {0:.2f}m".format(left_curverad) 
        cv2.putText(result, left_roc, (460,100), font, 1, (255,255,255), 2)
        right_roc = "Right RoC: {0:.2f}m".format(right_curverad) 
        cv2.putText(result, right_roc, (460,150), font, 1, (255,255,255), 2)
        # Calculate vehicle position relative to center
        # Calculate lane center
        lane_center = leftx[0] + (rightx[0] - leftx[0])/2
        # Assuming camera is positioned exactly in the middle of the car in the front
        rel_car_pos = (1280/2 - lane_center) * xm_per_pix
        # Output position
        rel_car_pos = "Dist from center: {0:.2f}m".format(rel_car_pos) 
        cv2.putText(result, rel_car_pos, (420,50), font, 1, (255,255,255), 2)
        return result
    else:
        print('Was not able to detect the lane')
        return(u_img)