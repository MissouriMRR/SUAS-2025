"""Takes the contour of an ODLC shape and determine which shape it is in the certain file path"""

import numpy as np
import scipy
from scipy import signal
from vision.common import constants as consts
from vision.common import odlc_characteristics as chars
from nptyping import NDArray, Shape, IntC, Float64
from vision.common.bounding_box import BoundingBox as bbox
from vision.common.bounding_box import tlwh_to_vertices
from vision.common.bounding_box import ObjectType
import json
from typing import List, Tuple


# constants

# Represents the number of intervals used when analyzing polar graph of shape
NUM_STEPS: int = 128

# Minimum prominence for a peak to be recognized on a shape contour
PROMINENCE: float = 0.05

# Lowered prominence to find all peaks of a cross
CROSS_PROMINENCE: float = 0.02

# Minimum allowed smallest radius to be considered a circle
MIN_CIRCLE_RADIUS: float = 0.9
# Minimum allowed difference between the shortest peaks of a quarter circle
MIN_DIFF_SHORT_PEAKS_QC: float = 0.15

# Maximum allowed difference between the longest peaks of a quarter circle
# Quarter Circle should have 2 long peaks which are very close in length, and one peak which is a bit shorter
MAX_DIFF_LONG_PEAKS_QC: float = 0.5

# Maximum allowed smallest radius for a shape to be considered a star
MAX_SMALLEST_RADIUS_STAR: float = 0.65

# The bottom percentage of a shape's polar graph that is removed to safely lower prominece and highlight the cross's slight peaks
PERCENT_CROSS_IGNORED: float = 0.85

# The maximum allowed average difference between points in compared shape to be considered the same type
MAX_ABS_ERROR: float = 1 / 8


# Keys are the number of peaks, targets are the cooresponding shape
shape_from_peaks = {
    1: chars.ODLCShape.CIRCLE,
    2: chars.ODLCShape.SEMICIRCLE,
    4: chars.ODLCShape.RECTANGLE,
    8: chars.ODLCShape.CROSS,
}

# Keys are ODLC shapes, and targets are their cooresponding order their sample arrays are stored in
ODLCShape_To_ODLC_Index = {
    chars.ODLCShape.CIRCLE: 0,
    chars.ODLCShape.QUARTER_CIRCLE: 1,
    chars.ODLCShape.SEMICIRCLE: 2,
    chars.ODLCShape.TRIANGLE: 3,
    chars.ODLCShape.PENTAGON: 4,
    chars.ODLCShape.STAR: 5,
    chars.ODLCShape.RECTANGLE: 6,
    chars.ODLCShape.CROSS: 7,
}






def process_shapes(
    contours: list[consts.Contour], hierarchy: consts.Hierarchy, image_dims: tuple[int, int]
) -> list[bbox]:
    """
    Takes all of the contours of an image and will return BoundingBox list w/ shape attributes

    Parameters
    ----------
    contours : list[consts.Contour]
        List of all contours from the image (from cv2.findContours())
        NOTE: cv2.findContours() returns as a tuple, so convert it to list w/ list(the_tuple)
    hierarchy : consts.Hierarchy
        The contour hierarchy information returned from cv2.findContours()
        (The 2nd returned value)
    image_dims : tuple[int, int]
        The dimensions of the image the contours are from.
        y_dim : int
            Image height.
        x_dim : int
            Image width.

    Returns
    -------
    bounding_boxes : list[bbox.BoundingBox]
        A list of BoundingBox objects that are the upright bounding box arround each corresponding
        contour at same index in list and with an attribute that is {"shape": chars.ODLCShape}
        with the identified shape or {"shape": None} if the contour does not match any.
    """
    bbox_list: List[bbox]
    for contour in contours:
        Shape_Type: chars.ODLCShape | None = classify_shape(contour)

        if not Shape_Type == None:
            min_y: int = contour[0][0][0]
            max_y: int = contour[0][0][0]
            min_x: int = contour[0][0][1]
            max_x: int = contour[0][0][1]
            vertices: consts.Corners

            list_of_points: NDArray[Shape["2,2"], IntC]
            for list_of_points in contour:
                point: Tuple[int, int] = list_of_points[0]
                y, x = point
                max_y = max(y, max_y)
                min_y = min(y, min_y)
                max_x = max(x, max_x)
                min_x = min(y, min_y)
            height: int = max_y - min_y
            width: int = max_x - min_x
            vertices = tlwh_to_vertices(min_x, max_y, width, height)

        bounding_box: bbox
        bbox.__init__(bounding_box, vertices, ObjectType.STD_OBJECT, None)
        bbox_list.append(bounding_box)
    return bbox_list


def classify_shape(contour: consts.Contour) -> chars.ODLCShape | None:
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
        if the given contour is not an ODLC shape (doesn't match any ODLC)
    """
    return compare_based_on_peaks(generate_polar_array(contour))


def compare_based_on_peaks(polar_array: NDArray[Shape["128"], Float64]) -> chars.ODLCShape | None:
    """
    Will determine if a polar array matches any ODLC shape, then verify that choice by comparing to sample

    Parameters
    ----------
    polar_array : NDArray[Shape["128"], Float64]
        An array storing the radii values of a normalized polar array

    Returns
    -------
    shape : chars.ODLCShape | None
        Will return one of the ODLC shapes defined in vision/common/odlc_characteristics or None
        if the given contour is not an ODLC shape (doesnt match any ODLC)
    """

    min_index: int

    # Normalizes radii to all be between 0 and 1
    polar_array /= np.max(polar_array)
    min_index = np.argmin(polar_array)

    # Rolls all values to put minimum radius at x = 0
    polar_array = np.roll(polar_array, -min_index)
    peaks: NDArray[Shape["*"], Float64]
    peaks = signal.find_peaks(polar_array, prominence=PROMINENCE)[0]
    num_peaks: int
    num_peaks = len(peaks)
    ODLC_guess: chars.ODLCShape

    # If the minimum value is greater than .9 (90% of Maximum Radius), then it is a circle
    if polar_array[0] > MIN_CIRCLE_RADIUS:
        ODLC_guess = chars.ODLCShape.CIRCLE

    # If we have a shape able to be uniquely defined by it's number of peaks
    elif num_peaks == 2 or num_peaks == 4 or num_peaks == 8:
        ODLC_guess = shape_from_peaks[num_peaks]

    # Must narrow down from triangle or quarter circle
    elif num_peaks == 3:
        # Sort peaks in by increasing value
        peaks = np.asarray(peaks)
        peaks_vals: NDArray[Shape["3"], Float64] = [0.0] * 3
        peaks_vals[0] = polar_array[peaks[0]]
        peaks_vals[1] = polar_array[peaks[1]]
        peaks_vals[2] = polar_array[peaks[2]]
        peaks_vals = np.sort(peaks_vals)
        # If there is a small enough difference between 2 greatest peaks,
        # And large enough difference between 2 smallest peaks, we have
        # A quarter cricle
        if (
            peaks_vals[2] - peaks_vals[1] < MAX_DIFF_LONG_PEAKS_QC
            and peaks_vals[1] - peaks_vals[0] > MIN_DIFF_SHORT_PEAKS_QC
        ):
            ODLC_guess = chars.ODLCShape.QUARTER_CIRCLE
        else:
            ODLC_guess = chars.ODLCShape.TRIANGLE

    # Must narrow down from pentagon or star
    elif num_peaks == 5:
        min = np.min(polar_array)
        # If minimum radius is less than .65 (65% of maximum radius), the we have a star
        if min < MAX_SMALLEST_RADIUS_STAR:
            ODLC_guess = chars.ODLCShape.STAR
        else:
            ODLC_guess = chars.ODLCShape.PENTAGON
    # This elif states that is the upper 15% of a shape has 8 peaks when prominence is decreased, it is likely a crosss
    # This was added because many crosses were not showing 2 peaks per "beam" in higher prominence, but rather those peaks were blending into one
    elif (
        len(
            signal.find_peaks(
                [0 if val < PERCENT_CROSS_IGNORED else val for val in polar_array],
                prominence=CROSS_PROMINENCE,
            )[0]
        )
        == 8
    ):
        ODLC_guess = chars.ODLCShape.CROSS
    else:
        return None

    # Read the appropriate Array from json file
    shape_json_address: str = "vision/standard_object/sample_ODLCs.json"
    with open(shape_json_address) as f:
        sample_shapes: NDArray[Shape["8, 128"], Float64] = json.load(f)

    # Finds the correct sample shape's array
    sample_shape: NDArray[Shape["128"], Float64] = sample_shapes[
        ODLCShape_To_ODLC_Index[ODLC_guess]
    ]
    sample_shape = np.asarray(sample_shape)
    if not verify_shape_choice(polar_array, sample_shape):
        return None
    return ODLC_guess


def generate_polar_array(cnt: consts.Contour) -> NDArray[Shape["128"], Float64]:
    """
    Generates 2 arrays storing the x and y coordinates of a new polar array

    Parameters
    ----------
    cnt : consts.Contour
        A contour returned by the contour detection algorithm

    Returns
    -------
    y: NDArray[Shape["128"], Float64]
        Will return one of the ODLC shapes defined in vision/common/odlc_characteristics or None
        if the given contour is not an ODLC shape (doesnt match any ODLC)
    """

    x_avg: Float64 = 0
    y_avg: Float64 = 0
    num_points: int = 0
    point: NDArray[Shape["2, 2"], IntC, Float64]
    # Finds average x and y value
    for point in cnt:
        y_avg += point[0][0]
        x_avg += point[0][1]
        num_points += 1

    x_avg = x_avg // num_points
    y_avg = y_avg // num_points
    i: int = 0
    # Centers the shape at (0,0) to allow for a better scan of the shape in polar coordinates
    for point in cnt:
        point[0][0] -= y_avg
        point[0][1] -= x_avg
        i += 1

    # Converts array of rectangular coordinates (x,y) to polar (angle, radius)
    pol_cnt: NDArray[Shape["*, 2"], Float64] = cartesian_array_to_polar(cnt)
    polar_sorted_indices = np.argsort(pol_cnt[:, 1])
    pol_cnt_sorted = pol_cnt[polar_sorted_indices]
    x: NDArray[Shape["*"], Float64]
    y: NDArray[Shape["*"], Float64]
    x, y = condense_polar(pol_cnt_sorted)

    return y


def condense_polar(
    polar_array: NDArray[Shape["*, 2"], Float64]
) -> NDArray[Shape["128,2"], Float64]:
    """
    Condenses a polar array to have 'NUM_STEPS' data points for analysis

    Parameters
    ----------
    polar_array : NDArray[Shape["*, 2"], Float64]
        An array of polar points of unknown length

    Returns
    -------
    new_x, new_y : NDArray[Shape["128"], Float64], NDArray[Shape["128"], Float64]

    """
    # Converting data to a form able to be passed into scipy.interp1d
    x: NDArray[Shape["*"]] = np.empty(len(polar_array))
    y: NDArray[Shape["*"]] = np.empty(len(polar_array))
    for i in range(len(polar_array)):
        x[i] = polar_array[i][1]
        y[i] = polar_array[i][0]

    # Linear interpolation to normalize all shapes to have the same number of data points
    new_x: NDArray[Shape["128"], Float64] = np.linspace(x.min(), x.max(), num=NUM_STEPS)
    new_y: NDArray[Shape["128"], Float64] = scipy.interpolate.interp1d(
        x, y, kind="linear", fill_value="extrapolate"
    )(new_x)
    return new_x, new_y




def cartesian_to_polar(x: float, y: float) -> tuple[float, float]:
    """
    Converts a rectangular (cartesian) coordinate to polar

    Parameters
    ----------
    x, y : float, float
        the x and y value of a point in rectangular (cartesian) coordinates

    Returns
    -------
    (rho, phi) : (float, float)
        Returns a tuple of the radius and angle of the cartesian point converted to polar coordinates
    """
    rho: float = np.sqrt(x**2 + y**2)
    phi: float = np.arctan2(y, x)
    polar_point: Tuple[float, float] = (rho, phi)
    return polar_point

def cartesian_array_to_polar(
    cartesian_array: NDArray[Shape["*,2,2"], Float64]
) -> NDArray[Shape["*,2"], Float64]:
    """
    Converts an array of rectangular (cartesian) coordinates to an array of polar coordinates

    Parameters
    ----------
    arr : List[float]
        An array of rectangular (cartesian) coordinates

    Returns
    -------
    shape : chars.ODLCShape | None
        Returns an array of polar coordinates
    """
    cartesian_array_x: NDArray[Shape["*, 2"], Float64] = cartesian_array[:, 0, 1]
    cartesian_array_y: NDArray[Shape["*, 2"], Float64] = cartesian_array[:, 0, 0]
    polar_array: NDArray[Shape["2, *"], Float64] = cartesian_to_polar(
        cartesian_array_x, cartesian_array_y
    )
    reformated_polar_array: NDArray[Shape["*, 2"], Float64] = np.swapaxes(polar_array, 0, 1)
    return reformated_polar_array


def verify_shape_choice(
    mystery_radii_list: NDArray[Shape["128"], Float64],
    sample_ODLC_radii: NDArray[Shape["128"], Float64],
) -> bool:
    """
    Verifies that an ODLC's Polar graph is adequately similar to the "guessed" ODLC's sample graph

    Parameters
    ----------
    mystery_radii_list : NDArray[Shape["128"], Float64]
        The list of radii of the ODLC shape we are trying to classify
    sample_ODLC_radii : NDArray[Shape["128"], Float64]
        The sample list of radii for the guessed ODLC shape

    Returns
    -------
    shape : bool
        Returns true if the absolute difference of the two radii lists is small enough, false if not
    """
    difference: float = np.sum(np.abs(mystery_radii_list - sample_ODLC_radii))
    return difference < NUM_STEPS * MAX_ABS_ERROR
