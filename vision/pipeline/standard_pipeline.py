"""Functions that perform standard object detection, localization, and classification"""

from typing import TypeAlias

import numpy as np

from nptyping import NDArray, Shape, UInt8, Float32
import vision.common.constants as consts

from vision.common.crop import crop_image
from vision.competition_inputs.bottle_reader import BottleData
from vision.common.bounding_box import BoundingBox
from vision.common.odlc_characteristics import ODLCColor

from vision.standard_object.odlc_contour_detection import fetch_shape_contours
from vision.standard_object.odlc_classify_shape import process_shapes
from vision.standard_object.odlc_text_detection import get_odlc_text
from vision.standard_object.odlc_colors import find_colors

import vision.pipeline.pipeline_utils as pipe_utils

ContourHeirarchyList: TypeAlias = list[tuple[tuple[consts.Contour, ...], consts.Hierarchy]]

# The various thresholds to run the image processing at
PROCESSING_THRESHOLDS: list[tuple[int, int]] = [(0, 50), (25, 150), (50, 250), (75, 350)]


def find_standard_objects(
    original_image: consts.Image, camera_parameters: consts.CameraParameters, image_path: str
) -> list[BoundingBox]:
    """
    Finds all bounding boxes of standard objects in an image

    Parameters
    ----------
    original_image: Image
        The image to find shapes in
    camera_parameters: CameraParameters
        The details of how and where the photo was taken
    image_path: str
        The path for the image the bounding box is from

    Returns
    -------
    found_odlcs: list[BoundingBox]
        The list of bounding boxes of detected standard objects
    """

    found_odlcs: list[BoundingBox] = []
    contours: list[consts.Contour] = fetch_shape_contours(original_image, True, "contours.jpg")
    shapes: list[BoundingBox] = process_shapes(contours)
    shape: BoundingBox
    for shape in shapes:
        # Set the shape attributes by reference. If successful, keep the shape
        if set_shape_attributes(shape, original_image) and pipe_utils.set_generic_attributes(
            shape, image_path, original_image.shape, camera_parameters
        ):
            found_odlcs.append(shape)

    return found_odlcs


def set_shape_attributes(
    shape: BoundingBox,
    original_image: consts.Image,
) -> bool:
    """
    Gets the attributes of a shape returned from process_shapes()
    Modifies `shape` in place

    Parameters
    ----------
    shape: BoundingBox
        The bounding box of the shape. Attribute "shape" must be set
    original_image: Image
        The image used to get the details for each shape

    Returns
    -------
    attributes_found: bool
        Returns true if all attributes were successfully found
    """

    if shape.get_attribute("shape") is None:
        return False

    odlc_img: consts.Image = crop_image(original_image, shape)

    text_bounding: BoundingBox = get_odlc_text(odlc_img)

    shape_color: ODLCColor
    text_color: ODLCColor

    if not text_bounding.get_attribute("text"):
        # No text was found, we can only get the shape color
        _, shape_color = find_colors(odlc_img)
        shape.set_attribute("shape_color", shape_color)
    else:
        # Text found, we can try to look for both colors
        shape.set_attribute("text", text_bounding.get_attribute("text"))
        text_img: consts.Image = crop_image(odlc_img, text_bounding)
        shape_color, text_color = find_colors(text_img)

        shape.set_attribute("shape_color", shape_color)
        shape.set_attribute("text_color", text_color)

    return True


def sort_odlcs(
    bottle_info: dict[str, BottleData], saved_odlcs: list[BoundingBox]
) -> list[list[BoundingBox]]:
    """
    Sorts the standard objects in the given list by which bottle they match

    Parameters
    ----------
    bottle_info: dict[str, BottleData]
        The data describing the object matching each bottle
    saved_odlcs: list[BoundingBox]
        The list of all sightings of standard objects

    Returns
    -------
    sorted_odlcs: list[list[BoundingBox]]
        The list of sightings of each object, matched to bottles
    """

    # The first index represents the bottle index - that's why there's 5
    sorted_odlcs: list[list[BoundingBox]] = [[], [], [], [], []]

    shape: BoundingBox
    for shape in saved_odlcs:
        bottle_index: int = get_bottle_index(shape, bottle_info)

        # Save the shape bounding box in its proper place
        if bottle_index != -1:
            sorted_odlcs[bottle_index].append(shape)

    return sorted_odlcs


def get_bottle_index(shape: BoundingBox, bottle_info: dict[str, BottleData]) -> int:
    """
    For the input ODLC BoundingBox, find the index of the bottle that it best matches.
    Returns -1 if no good match is found

    Parameters
    ----------
    shape: BoundingBox
        The bounding box of the shape. Attributes "text", "shape", "shape_color", and
        "text_color" must be set
    bottle_info: list[BottleData]
        The input info from bottle.json

    Returns
    -------
    bottle_index: int
        The index of the bottle from bottle.json that best matches the given ODLC
        Returns -1 if no good match is found
    """

    # For each of the given bottle shapes, find the number of characteristics the
    #   discovered ODLC shape has in common with it
    all_matches: NDArray[Shape[5], UInt8] = np.zeros((5), dtype=UInt8)

    index: str
    info: BottleData
    for index, info in bottle_info.items():
        matches: int = 0

        # if shape.get_attribute("text") == info["letter"]:
        #    matches += 1

        if shape.get_attribute("shape") == info["shape"]:
            matches += 1

        if shape.get_attribute("shape_color") == info["shape_color"]:
            matches += 1

        # if shape.get_attribute("text_color") == info["letter_color"]:
        #    matches += 1

        all_matches[int(index)] = matches

    # This if statement ensures that bad matches are ignored, and standards can be lowered.
    #   Still takes the best match, but if none are good enough they will be ignored.
    if all_matches.max() > 0:
        # Gets the index of the first bottle with the most matches.
        # First [0] takes the first dimension, second [0] takes the first element
        return np.where(all_matches == all_matches.max())[0][0]

    return -1


def add_emergent_object(
    odlc_dict: consts.ODLCDict, bottle_info: dict[str, BottleData], emg_object: BoundingBox
) -> consts.ODLCDict:
    """Adds the emergent object location to the given
    ODLC dictionary if one of the bottles is marked as
    associated to a emergent object

    Parameters
    ----------
    odlc_dict : consts.ODLCDict
        The dictionary of ODLCs matching the output format
    bottle_info : dict[str, BottleData]
        The data describing the object matching each bottle
    emg_object : BoundingBox
        The bounding box of the emergent object. Attributes "latitude" and "longitude"
        must be set

    Returns
    -------
    consts.ODLCDict
        The updated ODLC dictionary
    """
    bottle: tuple[str, BottleData]
    for bottle in bottle_info.items():
        if bottle[1]["shape"] == "Emergent":
            odlc_dict[bottle[0]] = {
                "latitude": emg_object.get_attribute("latitude"),
                "longitude": emg_object.get_attribute("longitude"),
            }
    return odlc_dict


def create_odlc_dict(sorted_odlcs: list[list[BoundingBox]]) -> consts.ODLCDict:
    """
    Creates the ODLC_Dict dictionary from a list of shape bounding boxes

    Parameters
    ----------
    sorted_odlcs: list[list[BoundingBox]]
        The list of sightings of each object, matched to bottles

    Returns
    -------
    odlc_dict: consts.ODLC_Dict
        The dictionary of ODLCs matching the output format
    """

    odlc_dict: consts.ODLCDict = {}

    i: int
    bottle: list[BoundingBox]
    for i, bottle in enumerate(sorted_odlcs):
        coords_list: list[tuple[int, int]] = []

        shape: BoundingBox
        for shape in bottle:
            coords_list.append((shape.get_attribute("latitude"), shape.get_attribute("longitude")))

        if len(bottle) > 0:
            coords_array: NDArray[Shape["*, 2"], Float32] = np.array(coords_list)

            average_coord: NDArray[Shape["2"], Float32] = np.average(coords_array, axis=0)

            odlc_dict[str(i)] = {"latitude": average_coord[0], "longitude": average_coord[1]}

    return odlc_dict
