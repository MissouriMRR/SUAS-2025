import cv2
import numpy as np

def fetchShapeContours(filename:str, draw_contours:bool=False, resulting_file_name:str="") -> list[np.ndarray]:
    """
    Detects the boundaries of potential shapes based on a pixel or region's
    brightness and saturation, finding only the darkest, brightest, and 
    most saturated portions of an image, and filters out shapes that are
    guaranteed not to be the shapes we're expecting to see. 

    Parameters
    ----------
    filename : str
        the name of the image to look for contours within
    draw_contours : bool
        True if we want to draw the contours and output the result

    Returns
    -------
    contours : list[numpy.ndarray]
        a list of all contours that could potentially be shapes we expect
        contour : numpy.ndarray
            an array of all points that make up the contour
    """

    img = cv2.imread(filename)
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) # converts image to HLS color format
    new_img = cv2.GaussianBlur(new_img, (5,5), sigmaX=0, sigmaY=0) # blurs HLS image

    img_brightness:np.ndarray
    img_saturation:np.ndarray

    img_brightness = np.array(new_img[:,:,1]) # reads lightness of image as 2D array
    img_saturation = np.array(new_img[:,:,2]) # reads saturation of image as 2D array

    # slightly blends blurred and unblurred images of same type, prioritizing blurred images
    img_brightness = cv2.addWeighted(new_img[:,:,1], 0.3, img_brightness, 0.7, 0)
    img_saturation = cv2.addWeighted(new_img[:,:,2], 0.3, img_saturation, 0.7, 0)

    avg_brt:float = np.average(img_brightness) # gets average brightness 

    white_thresh:np.ndarray
    black_thresh:np.ndarray
    saturation_thresh:np.ndarray

    # gets all values with a brightness greater than 195, less than 60, 
    # and with a saturation of greater than 50 (or 125 if the image is excessively dark)
    _, white_thresh = cv2.threshold(img_brightness, 195, 255, cv2.THRESH_BINARY)
    _, black_thresh = cv2.threshold(img_brightness, 60, 255, cv2.THRESH_BINARY_INV)
    _, saturation_thresh = cv2.threshold(img_saturation, (50 if avg_brt > 60 else 125), 255, cv2.THRESH_BINARY)
    
    # expands each threshold by 3 pixels in the x and y direction to merge those in close proximity to one another
    white_thresh = cv2.dilate(white_thresh, np.ones([3,3], np.uint8))
    black_thresh = cv2.dilate(black_thresh, np.ones([3,3], np.uint8))
    saturation_thresh = cv2.dilate(saturation_thresh, np.ones([3,3], np.uint8))

    contours:np.ndarray

    # finds the contour outlines of the combined thresholds
    contours, _ = cv2.findContours((white_thresh+black_thresh+saturation_thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    all_contours:list[np.ndarray] = []
    for a in contours:
        # gets rectangle bounding entire contour
        x,y,w,h = cv2.boundingRect(a) 
        area = float(cv2.contourArea(a))

        # calculates area inside contour in proportion to the area of the bounding rectangle
        proportional_area = area/(w*h) 
        aspect_ratio = max(float(w)/h,float(h)/w)

        # calculates solidity/rigidity of shape (shapes with rougher sides have lower solidity)
        solidity = area / (cv2.contourArea(cv2.convexHull(a))) 
        
        # saves the contour if the area is a reasonable size, reasonably close to a square, 
        # is not extremely small compared to its bounding box, and does not have very rough edges.
        if (10000 >= cv2.contourArea(a) >= 300) and (aspect_ratio <= 3) and (proportional_area >= 0.4) and (solidity >= 0.75) :
            all_contours.append(a)

    # draws contours and writes to output image file (but only if specified)
    if draw_contours:
        for a in all_contours:
            cv2.drawContours(img, [a], 0, (0,0,255), 2)
        cv2.imwrite(resulting_file_name, img)

    # returns a filtered list of contours
    return all_contours