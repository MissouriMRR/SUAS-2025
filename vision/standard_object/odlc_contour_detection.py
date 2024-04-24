# ELI'S CODE IT'S VERY COOL YIPPEE

import cv2
import numpy as np

def fetchShapeContours(filename:str, draw_contours:bool=False, resulting_file_name:str=""):
    blur_const = 11 # set constant for blurring images

    img = cv2.imread(filename)
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) # converts image to HLS color format

    img_brightness = np.array(new_img[:,:,1]) # reads lightness of image as 2D array
    img_saturation = np.array(new_img[:,:,2]) # reads saturation of image as 2D array

    # blurs lightness and saturation images
    img_brightness = cv2.GaussianBlur(img_brightness, (blur_const,blur_const), sigmaX=0, sigmaY=0) 
    img_saturation = cv2.GaussianBlur(img_saturation, (blur_const,blur_const), sigmaX=0, sigmaY=0) 

    # mixes blurred and unblurred images of same type
    img_brightness = cv2.addWeighted(new_img[:,:,1], 0.3, img_brightness, 0.7, 0)
    img_saturation = cv2.addWeighted(new_img[:,:,2], 0.3, img_saturation, 0.7, 0)

    avg_brt = np.average(img_brightness) # gets average brightness 

    saturation_threshold = 60 
    white_th = 255 - saturation_threshold

    _, white_thresh = cv2.threshold(img_brightness, white_th, 255, cv2.THRESH_BINARY)
    _, black_thresh = cv2.threshold(img_brightness, saturation_threshold, 255, cv2.THRESH_BINARY_INV)
    if (avg_brt < (saturation_threshold + 5)):
        black_thresh = 255 - black_thresh
    _, saturation_thresh = cv2.threshold(img_saturation, (50 if avg_brt > saturation_threshold else 125), 255, cv2.THRESH_BINARY)
    
    white_thresh = cv2.dilate(white_thresh, np.ones([3,3], np.uint8))
    black_thresh = cv2.dilate(black_thresh, np.ones([3,3], np.uint8))
    saturation_thresh = cv2.dilate(saturation_thresh, np.ones([3,3], np.uint8))

    contours, _ = cv2.findContours(white_thresh+black_thresh+saturation_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    all_contours = []
    for a in contours:
        x,y,w,h = cv2.boundingRect(a)
        area = float(cv2.contourArea(a))
        proportional_area = area/(w*h)
        aspect_ratio = max(float(w)/h,float(h)/w)
        solidity = area / (cv2.contourArea(cv2.convexHull(a)))
        if (10000 >= cv2.contourArea(a) >= 300) and (aspect_ratio <= 3) and (proportional_area >= 0.4) and (solidity >= 0.75) :
            all_contours.append(a)

    if draw_contours:
        for a in all_contours:
            cv2.drawContours(img, [a], 0, (0,0,255), 2)
        cv2.imwrite(resulting_file_name, img)

    return all_contours

for l in ['A','B','C','D','E','F']:
    fetchShapeContours(f"base_images/flight{l}.jpg", True, f"contour_results/flight{l}_test.jpg")