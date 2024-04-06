"""
Takes the contour of an ODLC shape and determine which shape it is
""" 
import numpy as np
import cv2
import scipy
from scipy import signal
from typing import List
from vision.common import constants as consts
from vision.common import odlc_characteristics as chars
import json


# constants

NUM_STEPS: int = 128
# Represents the number of intervals used when analyzing polar graph of shape

PROMINENCE: float = 0.05
# Minimum prominence for a peak to be recognized on a shape contour

CROSS_PROMINENCE: float = 0.02
# Lowered prominence to find all peaks of a cross

MIN_SMALLEST_RADIUS_CIRCLE: float = 0.9
# Minimum allowed smallest radius to be considered a circle

MIN_DIFF_BETWEEN_SHORT_PEAKS_QUARTER_CIRCLE: float = 0.15
# Minimum allowed difference between the shortest peaks of a quarter circle

MAX_DIFF_BETWEEN_LONG_PEAKS_QUARTER_CIRCLE: float = 0.5 
# Maximum allowed difference between the longest peaks of a quarter circle
# Quarter Circle should have 2 long peaks which are very close in length, and one peak which is a bit shorter

MAX_SMALLEST_RADIUS_STAR: float = 0.65
# Maximum allowed smallest radius for a shape to be considered a star

PERCENT_OF_CROSS_IGNORED_TO_LOWER_PROMINENCE: float = .85
# The bottom percentage of a shape's polar graph that is removed to safely lower prominece and highlight the cross's slight peaks


shape_from_peaks = {
    1 : chars.ODLCShape.CIRCLE,
    2 : chars.ODLCShape.SEMICIRCLE,
    4 : chars.ODLCShape.RECTANGLE,
    8 : chars.ODLCShape.CROSS
}

# Convertes x and y rectangular values to radius, angle tuples
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

# Converts an array of Rectangular Coordinates to Polar
def toPolar(arr):
    polar = [] # Stores an array of angles and radii as tuples (radius, angle)
    for i in range(len(arr)):
        coord = cart2pol(arr[i][0][1],arr[i][0][0])
        polar.append(coord)
    return polar

# Python implementation of merge sort algorithm, slightly edited to fit our array structure of tuples (sorted based on increasing angle).
def merge_sort(data: List[int]) -> List[int]:
    data_length = len(data)
    if data_length < 2:
        return data
    
    results = list()
    midpoint = data_length // 2
    
    lefts = merge_sort(data[:midpoint])
    rights = merge_sort(data[midpoint:])
    
    l = r = 0
    while l < len(lefts) and r < len(rights):
        if lefts[l][1] <= rights[r][1]:
            results.append(lefts[l])
            l += 1
        else:
            results.append(rights[r])
            r += 1
    
    if l < len(lefts):
        results += lefts[l:]
    elif r < len(rights):
        results += rights[r:]
        
    return results


# Condenses the array of polar coordinates to have 'NUM_STEPS' points stored for analysis
def condense_polar(polar_array) :
    step = 2 * 3.141592 / NUM_STEPS # distance between each x-value (angle)
    new_array = np.empty(NUM_STEPS, dtype=tuple)
    current_step = -3.141592 + step # start at -PI (plus step, because at exactly -PI there is no data)
    index = 0 # index of new_array
    length_polar_array = len(polar_array)
    sum = 0
    last_i = 0

    x = np.empty(len(polar_array))
    y = np.empty(len(polar_array))
    for i in range (len(polar_array)):
        x[i] = polar_array[i][1]
        y[i] = polar_array[i][0]

    # Linear interpolation to normalize all shapes to have the same number of data points
    f = scipy.interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
    newx = np.linspace(x.min(), x.max(), num=NUM_STEPS)
    newy = f(newx)
    return newx, newy


# Returns an array of tuples of the radii and angles of the contours from a given image address. Format: (radius, angle)
def Generate_Polar_Array(cnt: consts.Contour): # Returns an array of Polar Coordinate Tuples, sorted with increasing angles from -PI to PI
    x_avg = 0
    y_avg = 0
    numPoints = 0
    # Finds average x and y value
    for point in cnt:
        y_avg += point[0][0]
        x_avg += point[0][1]
        numPoints += 1
    
    x_avg = x_avg // numPoints
    y_avg = y_avg // numPoints
    i = 0
    # Centers the shape at (0,0) to allow for a better scan of the shape in polar coordinates
    for point in cnt:
        point[0][0] -= y_avg
        point[0][1] -= x_avg
        i += 1

    pol_cnt = toPolar(cnt) # Converts array of rectangular coordinates (x,y) to polar (angle, radius)
    pol_cnt_sorted = merge_sort(pol_cnt)
    x, y = condense_polar(pol_cnt_sorted)

    return x, y

ODLCShape_To_ODLC_Index = {
    chars.ODLCShape.CIRCLE : 0,
    chars.ODLCShape.QUARTER_CIRCLE : 1,
    chars.ODLCShape.SEMICIRCLE : 2,
    chars.ODLCShape.TRIANGLE : 3,
    chars.ODLCShape.PENTAGON : 4,
    chars.ODLCShape.STAR : 5,
    chars.ODLCShape.RECTANGLE : 6,
    chars.ODLCShape.CROSS : 7
}

# Checks to see if an ODLC's Polar graph is adequately similar to the "guessed" ODLC's sample graph
def Verify_Shape_Choice(mystery_radii_list, shape_choice, sample_ODLC_radii):
    difference = 0.0
    for i in range(NUM_STEPS):
        difference += abs(mystery_radii_list[i] - sample_ODLC_radii[i])
    return difference < NUM_STEPS / 8 # IMPORTANT -------------------THIS EQUATION IS MINIMALLY TESTED--------------------
        
def Compare_Based_On_Peaks(mysteryArr) -> chars.ODLCShape | None:  # Returns the name of the most similar ODLC object given an array of Polar Tuples (radius, angle)
        mysteryArr_x, mysteryArr_y = mysteryArr
        mysteryArr_y /= np.max(mysteryArr_y) # Normalizes radii to all be between 0 and 1
        mystery_min_index = np.argmin(mysteryArr_y)
        mysteryArr_y = np.roll(mysteryArr_y, -mystery_min_index) # Rolls all values to put minimum radius at x = 0


        peaks = signal.find_peaks(mysteryArr_y, prominence=PROMINENCE)[0]
        num_peaks = len(peaks)
        ODLC_guess: chars.ODLCShape


        if(mysteryArr_y[0] > MIN_SMALLEST_RADIUS_CIRCLE): # If the minimum value is greater than .9 (90% of Maximum Radius), then it is a circle
            ODLC_guess = chars.ODLCShape.CIRCLE

        elif num_peaks == 2 or num_peaks == 4 or num_peaks == 8: # If we have a shape able to be uniquely defined by it's number of peaks
            ODLC_guess = shape_from_peaks[num_peaks]
        
        elif num_peaks == 3: # Must narrow down from triangle or quarter circle
            # Sort peaks in by increasing value
            peaks = (np.asarray(peaks))
            peaksVals = [0.0] * 3
            peaksVals[0] = mysteryArr_y[peaks[0]]
            peaksVals[1] = mysteryArr_y[peaks[1]]
            peaksVals[2] = mysteryArr_y[peaks[2]]
            peaksVals = np.sort(peaksVals)
            if peaksVals[2] - peaksVals[1] < MAX_DIFF_BETWEEN_LONG_PEAKS_QUARTER_CIRCLE and peaksVals[1] - peaksVals[0] > MIN_DIFF_BETWEEN_SHORT_PEAKS_QUARTER_CIRCLE: # If there is a small enough difference between 2 greatest peaks,
                                                                                        # And large enough difference between 2 smallest peaks, we have
                                                                                        # A quarter cricle
                ODLC_guess = chars.ODLCShape.QUARTER_CIRCLE
            else:
                ODLC_guess = chars.ODLCShape.TRIANGLE
            
        elif num_peaks == 5: # Must narrow down from pentagon or star
            min = np.min(mysteryArr_y)
            if min < MAX_SMALLEST_RADIUS_STAR: # If minimum radius is less than .65 (65% of maximum radius), the we have a star
                ODLC_guess = chars.ODLCShape.STAR
            else:
                ODLC_guess = chars.ODLCShape.PENTAGON
        elif len(signal.find_peaks([0 if val < PERCENT_OF_CROSS_IGNORED_TO_LOWER_PROMINENCE else val for val in mysteryArr_y], prominence=CROSS_PROMINENCE)[0]) == 8:
            ODLC_guess = chars.ODLCShape.CROSS
        # This elif states that is the upper 15% of a shape has 8 peaks when prominence is decreased, it is likely a crosss
        # This was added because many crosses were not showing 2 peaks per "beam" in higher prominence, but rather those peaks were blending into one
        else:
            return None
        # Read the appropriate Array from json file
        f = open("vision/standard_object/sample_ODLCs.json")
        sample_shapes = json.load(f)
        sample_shape = sample_shapes[ODLCShape_To_ODLC_Index[ODLC_guess]]  # Finds the correct sample shape's array
        sample_shape = np.asarray(sample_shape)
        if not Verify_Shape_Choice(mysteryArr_y, ODLC_guess, sample_shape):
            return None
        return ODLC_guess

def classify_shape(
    contour: consts.Contour
) -> chars.ODLCShape | None:
    return Compare_Based_On_Peaks(Generate_Polar_Array(contour))
"""
    Will determine if the contour matches any ODLC shape, then verify that choice by comparing to sample

    Parameters
    ----------
    contour : consts.Contour
        A contour returned by the contour detection algorithm

    Returns
    -------
    shape : chars.ODLCShape | None
        Will return one of the ODLC shapes defined in vision/common/odlc_characteristics or None
        if the given contour is not an ODLC shape (doesnt match any ODLC)
"""