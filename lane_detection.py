import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import time

def get_sample_image():
    return cv.cvtColor(cv.imread('./img/Udacity/image008.jpg'), cv.COLOR_BGR2RGB)

def calibrate_camera(images, nx, ny, show_corners=False):
    """Get camera calibration parameters from a set of images.

    Args:
        images (array of images): array of the images used to calibrate
        nx (int): number of chess corners in x direction
        ny (int): number of chess corners in y direction
        show_corners (bool, optional): show images with the corners included. Defaults to False.

    Returns:
        double: return values of cv.calibrateCamera
        ??: camera matrix
        ??: distortion coefficients
        ??: rotation vectors
        ??: translation vectors
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (nx,ny), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            
            img_d = cv.drawChessboardCorners(img, (nx,ny), corners2, ret)
            if show_corners:
                plt.imshow(img_d)
                plt.title(fname)
                plt.show()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

images = glob.glob('./img/Udacity/calib/calibration*.jpg')
_, mtx, dist, _, _ = calibrate_camera(images, 9, 6, show_corners=False)

def undistort(image):
    """Undistorts an image using the camera calibration parameters.

    Args:
        image (OpenCV2 Image): image to undistort
        mtx (??): camera calibration matrix
        dist (??): ???

    Returns:
        OpenCV2 Image: undistorted image
    """
    return cv.undistort(image, mtx, dist, None, mtx)

def warp(image):
    """Warp an image to a top-down view.

    Args:
        image (OpenCV2 Image): the image to get the top-down view of

    Returns:
        OpenCV2 Image: the top-down view of the image
        ??: the transformation matrix
        ??: the inverse transformation matrix
    """
    img_size = (image.shape[1], image.shape[0]) # width x height
    offset = 200 # pixels
    src = np.float32([(701 ,459),  #top right
                      (1055,680),  #bottom right
                      (265 ,680),  #bottom left
                      (580 ,459)]) #top left

    # point array dst - destintation points close to source points in a rectangle
    
    dst = np.float32([(img_size[0]-offset,0), #top right
                      (img_size[0]-offset,img_size[1]), #bottom right
                      (img_size[0]-img_size[0]+offset,img_size[1]), #bottom left
                      (img_size[0]-img_size[0]+offset,0)]) #top left

    # use cv.getPerspectiveTransform() to get M, the transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    
    # use.cv.getPerspectiveTransform() to get Minv, the inverse transform matrix
    Minv = cv.getPerspectiveTransform(dst, src)
    
    # use cv.warpPerspective() to warp your image to a top-down view
    warped = cv.warpPerspective(image, M, img_size)

    return warped, M, Minv

def extract_lab_color_spaces(uwimg):
    unwarped_LAB = cv.cvtColor(uwimg, cv.COLOR_RGB2Lab)
    unwarp_L = unwarped_LAB[:,:,0]
    unwarp_A = unwarped_LAB[:,:,1]
    unwarp_B = unwarped_LAB[:,:,2]
    
    return unwarp_L, unwarp_A,unwarp_B

# Extract H,S, and L color channels from HLS color space.
def extract_hls_color_spaces(uwimg):
    unwarp_HLS = cv.cvtColor(uwimg, cv.COLOR_RGB2HLS)
    unwarp_HLS_H = unwarp_HLS[:, :, 0]
    unwarp_HLS_L = unwarp_HLS[:, :, 1]
    unwarp_HLS_S = unwarp_HLS[:, :, 2]
    
    return unwarp_HLS_H,unwarp_HLS_L,unwarp_HLS_S

# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_l_nomalize(img, thresh=(220, 255)):
    """
    This is used to nomalize HLS L color channel
    """
    # 1) Convert to HLS color space
    _,hls_l,_ = extract_hls_color_spaces(img)
    hls_l = hls_l*(255/np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def lab_b_nomalize(unwarped_img, thresh=(190,255)):
    """
    This is used to LAB B color channel
    """
    _,_,lab_b = extract_lab_color_spaces(unwarped_img)
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def filter_lanes2(image):
    # HLS L-channel Threshold (using default parameters)
    img_hls_L = hls_l_nomalize(image)

    # Lab B-channel Threshold (using default parameters)
    img_lab_B = lab_b_nomalize(image)
    
    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_lab_B)
    combined[(img_hls_L == 1) | (img_lab_B == 1)] = 1

    # reduce lines to 1px width
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    perwitt = cv.filter2D(combined, -1, kernel)
    threshold = cv.threshold(perwitt, 0, 255, cv.THRESH_BINARY)[1]
    
    return threshold

def find_lane_points(image):
    width = image.shape[1]
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    for y, x in np.argwhere(image > 0):
        if x < width/2:
            x_left.append(x)
            y_left.append(y)
        else: 
            x_right.append(x)
            y_right.append(y)

    return x_left, y_left, x_right, y_right

def fit_polynomial(x_coordinates, y_coordinates):
    """Fit a second order polynomial to the given coordinates.

    Args:
        x_coordinates (int[]): x coordinates of the lane points
        y_coordinates (int[]): y coordinates of the lane points

    Returns:
        float[]: array of polynomial coefficients (length 3)
    """
    return np.polyfit(y_coordinates, x_coordinates, 2)

def draw_lane(height, width, w_left, w_right):
    """Draw the lane on a black image.

    Args:
        height (int): height of the image
        width (int): width of the image
        w_left (float[]): polynomial coefficients of the left lane
        w_right (float[]): polynomial coefficients of the right lane

    Returns:
        OpenCV2 Image: Image with drawn lane and lane markings
    """
    lanes = np.zeros((height,width,3), np.uint8)

    x_scale = np.linspace(0, height, 100)
    y_left = np.polyval(w_left, x_scale)
    y_right = np.polyval(w_right, x_scale)

    left_points = (np.asarray([y_left, x_scale]).T).astype(np.int32)
    right_points = (np.asarray([y_right, x_scale]).T).astype(np.int32)

    cv.polylines(lanes, [left_points], False, (18, 102, 226), thickness=15)
    cv.polylines(lanes, [right_points], False, (18, 102, 226), thickness=15)
    
    spur = np.concatenate((left_points, right_points[::-1]))
    cv.fillPoly(lanes, np.int32([spur]), (0,255,0))

    return lanes

def get_radius(w, width):
    """Calculate the radius of the lane.

    Args:
        w (float[]): polynomial coefficients of the lane

    Returns:
        float: radius of the lane
    """
    # TODO: muss das hier nicht width sein?: nein wil hier jemand width und height vertauscht hat
    return ((1 + (2*w[0]*width + w[1])**2)**1.5) / np.absolute(2*w[0])

def get_kruemmung(w_left, w_right, width):
    """Calculate the curvature of the lane. Averages the curvature of the left and right lane.

    Args:
        w_left (float[]): polynomial coefficients of the left lane
        w_right (float[]): polynomial coefficients of the right lane

    Returns:
        float: curvature of the lane
    """
    rad_left = get_radius(w_left, width)
    rad_right = get_radius(w_right, width)
    return (rad_left + rad_right) / 2
    
def unwarp(img, M, Minv):
    warped = cv.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR)
    return warped

def overlay(original, overlay, opacity=0.5):
    """
    This function overlays the detected lane area onto the original image
    """
    # Make sure the overlay and original images are in the same size
    assert original.shape == overlay.shape
    
    # Create a copy of the original image
    result = original.copy()
    
    # Overlay the mask onto the original image
    result = cv.addWeighted(original, 1.0, overlay, opacity, 0)
    
    return result

def pipeline(image, show_debug=False):
    """Pipeline for lane detection.

    Args:
        image (OpenCV2 Image): Image to process
        show_debug (bool, optional): Whether to show debug images in the Jupyter Notebook. Defaults to False.

    Returns:
        OpenCV2 Image: Image with detected lane markings
    """
    # undistort

    last = time.time()
    undistorted = undistort(image)
    if show_debug:
        plt.imshow(undistorted)
        plt.title("Undistorted image")
        plt.show()

    print("undistort", time.time()- last)
    last = time.time()

    # Persfective transform
    warped, M, Minv = warp(undistorted)
    if show_debug:
        plt.imshow(warped)
        plt.title("Warped image")
        plt.show()

    
    print("warp", time.time()- last)
    last = time.time()

    # Filter lanes
    filtered = filter_lanes2(warped)
    if show_debug:
        plt.imshow(filtered, cmap='gray')
        plt.title("Filtered image")
        plt.show()

    
    print("Filter", time.time()- last)
    last = time.time()

    # Spurpunkte finden
    x_left, y_left, x_right, y_right = find_lane_points(filtered)
    w_left = fit_polynomial(x_left, y_left)
    w_right = fit_polynomial(x_right, y_right)


    print("Poly", time.time()- last)
    last = time.time()

    # Kurze einzeichnen
    width, height = filtered.shape
    lanes = draw_lane(width, height, w_left, w_right)
    if show_debug:
        plt.imshow(lanes)
        plt.title("Spurmarkierungen")
        plt.show()

    
    print("draw", time.time()- last)
    last = time.time()

    # zurück-warpen
    unwarped = unwarp(lanes, M, Minv)
    if show_debug:
        plt.imshow(unwarped)
        plt.title("Spur zurücktransformiert")
        plt.show()

    
    print("unwarp", time.time()- last)
    last = time.time()

    # Überlagern
    result = overlay(undistorted, unwarped)
    if show_debug:
        plt.imshow(result)
        plt.title("Spur überlagert")
        plt.show()

    krumm = get_kruemmung(w_left, w_right, width)
    if show_debug:
        print("Krümmung: ", krumm)

    cv.putText(result, "Radius: {:.2f}".format(krumm), (50, 50), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

    return result

input_video = cv.VideoCapture('img/Udacity/project_video.mp4')
 
# Check if camera opened successfully
if (input_video.isOpened()== False): 
  print("Error opening video stream or file")
else:
  v_width  = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
  v_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  fps = input_video.get(cv.CAP_PROP_FPS)
  print("Video size: ", v_width, "x", v_height)
  fourcc = cv.VideoWriter_fourcc(*'X264')
  output_video = cv.VideoWriter('output.mp4', fourcc, fps, (v_width, v_height))

 
# Read until video is completed
while(input_video.isOpened()):
  # Capture frame-by-frame
  ret, frame = input_video.read()
  if ret == True:
    cv.startWindowThread()

    # covert to rgb as pipeline expects rgb
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    start = time.time()
    # EIGENTLICHE PIPELINE
    output = pipeline(frame)
    print("Pipeline", time.time()-start, "\n\n\n\n\n")

    # convert back to bgr for saving video
    output = cv.cvtColor(output, cv.COLOR_RGB2BGR)
 
    # Display the resulting frame
    cv.imshow('Frame', output)
    
    # show single frame in notebook
    # plt.imshow(output[:,:,::-1])
    # plt.show()
    # break

    # write frame to video
    output_video.write(output)
 
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
input_video.release()
output_video.release()

# Closes all the frames
cv.destroyAllWindows()
cv.waitKey(1)