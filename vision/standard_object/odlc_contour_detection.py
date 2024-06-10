"""Implements contour filtering to return contours that are likely to be shapes we expect to see."""

# pylint: disable=too-many-locals

import cv2
import numpy as np

from vision.common.constants import Contour, Hierarchy, Image

# Contour restriction constants
MAX_CONTOUR_AREA: int = 10000
MIN_CONTOUR_AREA: int = 300
MAX_ASPECT_RATIO: float = 3
MIN_SOLIDITY: float = 0.75
MIN_PROPORTIONAL_AREA: float = 0.4

# Contour detection constants
KERNEL_SIZE: cv2.typing.Size = (5, 5)
MIN_WHITE_VALUE: int = 195
MAX_BLACK_VALUE: int = 60
MIN_SATURATION_VALUE: int = 50
MIN_DARK_SATURATION_VALUE: int = 125
BLUR_IMG_WEIGHT: float = 0.7
NORMAL_IMG_WEIGHT: float = 0.3


def fetch_shape_contours(
    image: Image, draw_contours: bool = False, resulting_file_name: str = ""
) -> list[np.ndarray[np.dtype[np.uint8], np.dtype[np.uint8]]]:
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

    img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # converts image to HLS color format

    # Blur is added to the image to reduce noise and to make the image/grass more uniform
    new_img = cv2.GaussianBlur(img, KERNEL_SIZE, sigmaX=0, sigmaY=0)  # blurs HLS image

    img_brightness: np.ndarray[np.dtype[np.uint8], np.dtype[np.uint8]]
    img_saturation: np.ndarray[np.dtype[np.uint8], np.dtype[np.uint8]]

    img_brightness = np.array(new_img[:, :, 1])  # reads lightness of image as 2D array
    img_saturation = np.array(new_img[:, :, 2])  # reads saturation of image as 2D array

    # slightly blends blurred and unblurred images of same type, prioritizing blurred images
    img_brightness = cv2.addWeighted(
        img[:, :, 1], NORMAL_IMG_WEIGHT, img_brightness, BLUR_IMG_WEIGHT, 0
    )
    img_saturation = cv2.addWeighted(
        img[:, :, 2], NORMAL_IMG_WEIGHT, img_saturation, BLUR_IMG_WEIGHT, 0
    )

    avg_brt: float = float(np.average(img_brightness))  # gets average brightness

    white_thresh: np.ndarray[np.dtype[np.uint8], np.dtype[np.uint8]]
    black_thresh: np.ndarray[np.dtype[np.uint8], np.dtype[np.uint8]]
    saturation_thresh: np.ndarray[np.dtype[np.uint8], np.dtype[np.uint8]]

    # Use higher saturation value if image is bright, lower if dark
    saturation_value: int = MIN_SATURATION_VALUE if avg_brt > 60 else MIN_DARK_SATURATION_VALUE

    # White threshold is used to find the brightest parts of the image
    # Black threshold is used to find the darkest parts of the image
    # Saturation threshold is used to find the most saturated parts of the image
    _, white_thresh = cv2.threshold(img_brightness, MIN_WHITE_VALUE, 255, cv2.THRESH_BINARY)
    _, black_thresh = cv2.threshold(img_brightness, MAX_BLACK_VALUE, 255, cv2.THRESH_BINARY_INV)
    _, saturation_thresh = cv2.threshold(img_saturation, saturation_value, 255, cv2.THRESH_BINARY)

    # expands each threshold by 3 pixels in the x and y direction
    # to merge those in close proximity to one another
    white_thresh = cv2.dilate(white_thresh, np.ones([3, 3], np.uint8))
    black_thresh = cv2.dilate(black_thresh, np.ones([3, 3], np.uint8))
    saturation_thresh = cv2.dilate(saturation_thresh, np.ones([3, 3], np.uint8))

    contours: list[Contour]
    _hierarchy: Hierarchy

    # finds the contour outlines of the combined thresholds
    contours, _hierarchy = cv2.findContours(
        (white_thresh + black_thresh + saturation_thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    all_contours: list[Contour] = []
    for contour in contours:
        # gets rectangle bounding entire contour
        _x_pos, _y_pos, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))

        # calculates area inside contour in proportion to the area of the bounding rectangle
        proportional_area = area / (width * height)
        aspect_ratio = max(float(width) / height, float(height) / width)

        # calculates solidity/rigidity of shape (shapes with rougher sides have lower solidity)
        solidity = area / (cv2.contourArea(cv2.convexHull(contour)))

        # saves the contour if the area is a reasonable size, reasonably close to a square,
        # is not extremely small compared to its bounding box, and does not have very rough edges.
        if (
            (MAX_CONTOUR_AREA >= cv2.contourArea(contour) >= MIN_CONTOUR_AREA)
            and (aspect_ratio <= MAX_ASPECT_RATIO)
            and (proportional_area >= MIN_PROPORTIONAL_AREA)
            and (solidity >= MIN_SOLIDITY)
        ):
            all_contours.append(contour)

    # draws contours and writes to output image file (but only if specified)
    if draw_contours:
        for contour in all_contours:
            cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)
        cv2.imwrite(resulting_file_name, image)

    # returns a filtered list of contours
    return all_contours
