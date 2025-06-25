"""Functions for calculating coordinate degree lengths"""

import math
import logging

from geographiclib.constants import Constants
from geographiclib.geodesic import Geodesic
import numpy as np
from vision.common.constants import FEET_PER_METER, CameraParameters, Vector
from vision.deskew import vector_utils


def latitude_length(latitude_deg: float) -> float:
    """
    Returns the distance in feet of one degree of latitude at a particular latitude

    Parameters
    ---------
    latitude_deg : float
        The latitude in degrees

    Returns
    -------
    latitude_length
        The length of a degree of latitude in feet at the given latitude

    References
    ----------
    https://en.wikipedia.org/wiki/Geographic_coordinate_system#Length_of_a_degree
    """

    # Convert to radians for trig functions
    latitude_rad: float = np.deg2rad(latitude_deg)

    # Formula is adapted from the referenced Wikipedia page
    distance: float = (
        111132.92
        - 559.82 * np.cos(2 * latitude_rad)
        + 1.175 * np.cos(4 * latitude_rad)
        - 0.0023 * np.cos(6 * latitude_rad)
    ) * FEET_PER_METER

    return distance


def longitude_length(latitude_deg: float) -> float:
    """
    Calculates the distance in feet of one degree of longitude at that latitude

    Parameters
    ---------
    latitude_deg : float
        The latitude in degrees

    Returns
    -------
    longitude_length
        The length of a degree of longitude in feet at the given latitude

    References
    ----------
    https://en.wikipedia.org/wiki/Geographic_coordinate_system#Length_of_a_degree
    """

    # Convert degrees to radians for trig functions
    latitude_rad: float = np.deg2rad(latitude_deg)

    # Formula is adapted from the referenced Wikipedia page
    distance: float = (
        111412.84 * np.cos(latitude_rad)
        - 93.5 * np.cos(3 * latitude_rad)
        + 0.118 * np.cos(5 * latitude_rad)
    ) * FEET_PER_METER

    return distance


def get_coordinates(
    pixel: tuple[int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    camera_parameters: CameraParameters,
) -> tuple[float, float] | None:
    """
    Calculates the coordinates of the given pixel.
    Returns None if there is no valid intersect.

    Parameters
    ----------
    pixel: tuple[int, int]
        The coordinates of the pixel in [X, Y] form
    image_shape : tuple[int, int, int] | tuple[int, int]
        The shape of the image (returned by `image.shape` when image is a numpy image array)
    camera_parameters: CameraParameters
        The details on how and where the photo was taken
        rotation_deg: list[float]
            The rotation of the drone in degrees.
        drone_coordinates: list[float]
            The coordinates of the drone in degrees of (latitude, longitude)
        altitude: float
            The altitude of the drone in meters

    Returns
    -------
    pixel_coordinates : tuple[float, float] | None
        The (latitude, longitude) coordinates of the pixel in degrees.
        Equal to None if there is no valid intersect.
    """
    geod: Geodesic = Geodesic(Constants.WGS84_a, Constants.WGS84_f)
    # Get the vector that points to the pixel
    # This will have a negative z since it is pointing at the projected image on the xy plane
    vector: Vector = vector_utils.pixel_vector(pixel, image_shape)

    # We want just the roll and pitch of the photo, without the pitch
    # introduced from the gimbal pointing down
    rotations: list[float] = [
        camera_parameters["rotation_deg"][0],
        camera_parameters["rotation_deg"][1] + 90,
        0,
    ]

    # Apply the rotation of the drone to the vector
    vector = vector_utils.rotate_degrees(vector, rotations)

    # Multiply the vector to make the z axis match the altitude of the drone
    vector = abs(camera_parameters["altitude"] / vector[2]) * vector

    hypotenuse: float = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    angle: float = math.degrees(math.atan2(vector[0], vector[1]))
    logging.debug("hypotenuse: %s", hypotenuse)
    logging.debug("angle: %s", angle)

    # Take into account both the yaw of the drone and the angle that the
    # pixel is at from the center of the image
    azimuth = camera_parameters["rotation_deg"][2] + angle  # in degrees
    shift = hypotenuse  # in meters

    logging.debug("azimuth: %s", azimuth)

    geod_result = geod.Direct(
        camera_parameters["drone_coordinates"][0],
        camera_parameters["drone_coordinates"][1],
        azimuth,
        shift,
    )

    return geod_result["lat2"], geod_result["lon2"]
