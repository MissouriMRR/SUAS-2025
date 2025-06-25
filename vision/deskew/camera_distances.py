"""Functions for calculating locations and distances of objects in an image"""

import numpy as np

from vision.common.constants import Corners, Point, CameraParameters, FEET_PER_METER
from vision.common.bounding_box import BoundingBox

from vision.deskew import coordinate_lengths
from vision.deskew import vector_utils
from vision.deskew.deskew import get_corner_points
from flight.waypoint.calculate_distance import calculate_distance as coordinate_calculate_distance


def corner_coords(image_shape, camera_parameters) -> Corners:

    coordinate_list = [
        coordinate_lengths.get_coordinates(point, image_shape, camera_parameters)
        for point in get_corner_points(image_shape)
    ]
    for coordinate in coordinate_list:
        if coordinate == None:
            raise ValueError(
                "One or more of the coordinates are zero, ensure that the image rotation is downward"
            )
    coords: Corners = np.array(
        coordinate_list,
        dtype=np.float64,
    )

    return coords


def bounding_area(
    box: BoundingBox,
    image_shape: tuple[int, int, int] | tuple[int, int],
    camera_parameters: CameraParameters,
) -> float | None:
    """
    Calculates the area in feet of the bounding box on the ground

    Parameters
    ----------
    box: BoundingBox
        The bounding box of the object
    image_shape : tuple[int, int, int] | tuple[int, int]
        The shape of the image (returned by `image.shape` when image is a numpy image array)
    camera_parameters: CameraParameters
        The details on how and where the photo was taken
        rotation_deg: list[float]
            The rotation of the drone in degrees. The constant ROTATION_OFFSET of the
            camera, stored in constants.py, will be applied first
        drone_coordinates: list[float]
            The coordinates of the drone. Not used in this function.
        altitude: float
            The altitude of the drone in meters

    Returns
    -------
    area : float | None
        The area of the bounding box in feet.
        Returns None if one or both of the points did not have an intersection
    """

    # Calculate the distance from the top left vertex to the top right vertex
    width_length: float | None = calculate_distance(
        box.vertices[0], box.vertices[1], image_shape, camera_parameters
    )

    # Calculate the distance from the top left vertex to the bottom left vertex
    height_length: float | None = calculate_distance(
        box.vertices[0], box.vertices[3], image_shape, camera_parameters
    )

    if height_length is None or width_length is None:
        return None

    return width_length * height_length


def calculate_distance(
    pixel1: tuple[int, int],
    pixel2: tuple[int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    camera_parameters: CameraParameters,
) -> float | None:
    """
    Calculates the physical distance between two points on the ground represented by pixel
    locations. Units of `distance` will be in feet

    Parameters
    ----------
    pixel1, pixel2: tuple[int, int]
        The two input pixel locations in [X,Y] form. The distance between them will be calculated

    image_shape : tuple[int, int, int] | tuple[int, int]
        The shape of the image (returned by `image.shape` when image is a numpy image array)
    camera_parameters: CameraParameters
        The details on how and where the photo was taken
        rotation_deg: list[float]
            The rotation of the drone in degrees. The constant ROTATION_OFFSET of the
            camera, stored in constants.py, will be applied first
        drone_coordinates: list[float]
            The coordinates of the drone. Not used in this function.
        altitude: float
            The altitude of the drone in meters

    Returns
    -------
    distance : float | None
        The distance between the two pixels. Units are the same units as `altitude`
        Returns None if one or both of the points did not have an intersection
    """

    # Convert meters to feet
    altitude = camera_parameters["altitude"] * 3.28084

    intersect1: Point | None = vector_utils.pixel_intersect(
        pixel1,
        image_shape,
        camera_parameters["rotation_deg"],
        altitude,
    )

    intersect2: Point | None = vector_utils.pixel_intersect(
        pixel2,
        image_shape,
        camera_parameters["rotation_deg"],
        altitude,
    )

    # Checks if the intersects were valid
    if intersect1 is None or intersect2 is None:
        return None

    # Calculate the distance between the two intersects
    distance: float = float(np.linalg.norm(intersect1 - intersect2))

    return distance


def pixel_per_foot(
    image_shape: tuple[int, int, int] | tuple[int, int], camera_parameters: CameraParameters
) -> float:
    Corner_list: Corners = corner_coords(image_shape, camera_parameters)

    # using this calculate distance instead so it will use coordinates instead of having to parse the image data makes it a little faster
    # returning in feet pls change this at some point
    # uisng the top left and top right parts of the image
    return (
        image_shape[1]
        / (
            (np.tan(vector_utils.get_fov()[0] / 2))
            * FEET_PER_METER
            * camera_parameters["altitude"]
            * 2
        )
        * 0.4
    )
    # print(coordinate_calculate_distance(Corner_list[0][0], Corner_list[0][1], 0, Corner_list[1][0], Corner_list[1][1], 0))
    # return (image_shape[1]/(FEET_PER_METER*coordinate_calculate_distance(Corner_list[0][0], Corner_list[0][1], 0, Corner_list[1][0], Corner_list[1][1], 0)))
