"""Functions that use vectors to calculate camera intersections with the ground"""

import math
import json

import numpy as np
from scipy.spatial.transform import Rotation

from utils import type_utils
from vision.common.constants import Point, Vector, CameraConfig


def pixel_intersect(
    pixel: tuple[int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    rotation_deg: list[float],
    height: float,
) -> Point | None:
    """
    Finds the intersection [X,Y] of a given pixel with the ground relative to the camera.
    A camera with no rotation points in the +X direction and is centered at [0, 0, height].

    Parameters
    ----------
    pixel : tuple[int, int]
        The location of the pixel in [X, Y] form
    image_shape : tuple[int, int, int] | tuple[int, int]
        The shape of the image (returned by image.shape when image is a numpy image array)
    rotation_deg : list[float]
        The [roll, pitch, yaw] rotation of the camera in degrees
    height : float
        The height that the image was taken at. The units of the output will be the units of the
        input.

    Returns
    -------
    intersect : Point | None
        The coordinates/distance [X,Y,Z] where the pixel's vector intersects with the ground. Units
            are the same as the units of `height`
        Returns None if there is no intersect.
    """

    # Get the vector that points to the pixel
    # This will have a negative z since it is pointing at the projected image on the xy plane
    vector: Vector = pixel_vector(pixel, image_shape)

    # Fix the pitch of the photo due to gimbal pointing down
    gimbal_rotation: list[float] = [rotation_deg[0], rotation_deg[1] + 90, rotation_deg[2]]

    # Apply the rotation of the drone to the vector
    vector = rotate_degrees(vector, gimbal_rotation)

    # This needs to be applied as unlike the usual x and y axis, y is flipped
    # This should only be applied in pixel_intersect and not get_coordinates!!!
    vector[1] *= -1

    intersect: Point | None = plane_collision(vector, height)

    return intersect


def plane_collision(ray_direction: Vector, height: float, cutoff_ratio: float = 0) -> Point | None:
    """
    Returns the vector where where the vector intersects with the plane
    The input vector is in XYZ and returns and XYZ vector with z equal to the height
    Returns None if there is no intersect.

    Parameters
    ----------
    ray_direction : Vector
        XYZ coordinates that represent the direction a ray faces
    height : float
        The Z coordinate for the starting height of the ray; can be any units
    cutoff_ratio: float = 0
        basically the ratio of z in the xyz vector where any vector less than it will get cutoff
        Not essential, but is used to prevent warping and cutoff nearing no height of the image
        aka they are getting closer to infinity horizontal distance for the vertical
        Ratio is roughly(not exact) if .1 then distance to the corner is 10 times as
        large as the distance to the camera and lower ratio means less quality of pixel
        default is 0 to allow near infinite intersects with the plane

    Returns
    -------
    intersect : Point | None
        The ray's intersection with the plane in [X,Y,Z] format.
        Z is the height and Units are the same as `height`
        Returns None if there is no intersect.
    """

    # negative Z is down towards the ground I think,
    # so as long as Z is negative then the vector is valid
    # however if the Z is so small that it is negligible is probably
    # not the best due to warping and annoyance
    # like a vector at 89 degrees from the plane will not give good results at all

    # normalize cause this how that goes
    intersect = ray_direction / np.linalg.norm(ray_direction)
    intersect = np.round(intersect, decimals=6)

    # Checks if the ray intersects with the plane - negative `time` means the intersection
    #   is behind the camera

    if intersect[2] > (cutoff_ratio * -1) or np.isnan(intersect[2]) or intersect[2] is None:
        return None

    # if no height to the image the camera is in plane so instanetaneous intersecting
    if intersect[2] == 0:
        return np.array([0, 0, 0], dtype=np.float64)

    # sets the z of the vector equal to negative height and scales the rest of the vector with it
    intersect = abs(height / intersect[2]) * intersect
    intersect = np.round(intersect, decimals=6)

    return intersect


def pixel_vector(
    pixel: tuple[int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> Vector:
    """
    Generates a vector representing the given pixel.
    Pixels are in row-major form [X, Y]

    Parameters
    ----------
    pixel : tuple[int, int]
        The pixel location in [X, Y] form
    image_shape : tuple[int, int, int] | tuple[int, int]
        The shape of the image (returned by image.shape when image is a numpy image array)

    Returns
    -------
    pixel_vector : Vector
        The vector that represents the direction of the given pixel
    """

    # Find the FOVs using the focal length
    fov_h: float
    fov_v: float
    fov_h, fov_v = get_fov()

    vector: Vector = camera_vector(
        pixel_angle(fov_h, pixel[0] / image_shape[1]),
        pixel_angle(fov_v, pixel[1] / image_shape[0]),
    )

    return vector


def pixel_angle(fov: float, ratio: float) -> float:
    """
    Calculates a pixel's angle from the center of the camera on a single axis. Analogous to the
    pixel's "fov"

    Only one component of the pixel is used here, call this function for each X and Y

    Parameters
    ----------
    fov : float
        The field of view of the camera in radians olong a given axis
    ratio : float
        The pixel's position as a ratio of the coordinate to the length of the image
        Example: For an image that is 1080 pixels wide, a pixel at position 270 would have a
        ratio of 0.25

    Returns
    -------
    angle : float
        The pixel's angle from the center of the camera along a single axis.
        The horizontal angle will be reversed
    """

    return np.arctan(np.tan(fov / 2) * (1 - 2 * ratio))


def get_fov() -> tuple[float, float]:
    """
    gets the focal length for the current drone

    Returns
    -------
    fov : tuple[float,float]
        The horizontal and vertical field of view in radians
    """

    with open("vision/common/camera_config.json", encoding="ascii") as file:
        camera_config: CameraConfig = json.load(file)
        h_fov: float
        v_fov: float
        if camera_config["airsim_flag"]:
            if camera_config["Airsim"]["horizontalFOV"] != 0:
                h_fov = camera_config["Airsim"]["horizontalFOV"]
            else:
                h_fov = calculate_fov("Airsim", "horizontalFOV")

            if camera_config["Airsim"]["verticalFOV"] != 0:
                v_fov = camera_config["Airsim"]["verticalFOV"]
            else:
                v_fov = calculate_fov("Airsim", "verticalFOV")
        else:
            if camera_config["Default"]["horizontalFOV"] != 0:
                h_fov = camera_config["Default"]["horizontalFOV"]
            else:
                h_fov = calculate_fov("Default", "horizontalFOV")

            if camera_config["Default"]["verticalFOV"] != 0:
                v_fov = camera_config["Default"]["verticalFOV"]
            else:
                v_fov = calculate_fov("Default", "verticalFOV")

    return h_fov, v_fov


def calculate_fov(camera: str, fov_type: str) -> float:
    """
    Converts a given focal length and sensor length to the corresponding field of view in
    radians stored in camera_config.json

    Parameters
    ----------
    camera : str
        The camera to calculate FOV for that is stored in camera_config ("Airsim" or "Default")
    fov_type : str
        Whichever FOV is needed to be calculated ("horizontalFOV" or "verticalFOV")
    """
    with open("vision/common/camera_config.json", encoding="ascii") as file:
        camera_config: CameraConfig = json.load(file)
        fov: float
        if camera == "Airsim":
            if fov_type == "horizontalFOV":
                camera_config["Airsim"]["horizontalFOV"] = 2 * np.arctan(
                    camera_config["Airsim"]["sensorWidth"]
                    / (2 * camera_config["Airsim"]["focal_length"])
                )
            if fov_type == "verticalFOV":
                camera_config["Airsim"]["verticalFOV"] = 2 * np.arctan(
                    camera_config["Airsim"]["sensorHeight"]
                    / (2 * camera_config["Airsim"]["focal_length"])
                )
            json.dump(camera_config, file)
            fov = camera_config["Airsim"][fov_type]
        else:
            if fov_type == "horizontalFOV":
                camera_config["Default"]["horizontalFOV"] = 2 * np.arctan(
                    camera_config["Default"]["sensorWidth"]
                    / (2 * camera_config["Default"]["focal_length"])
                )
            if fov_type == "verticalFOV":
                camera_config["Default"]["verticalFOV"] = 2 * np.arctan(
                    camera_config["Default"]["sensorHeight"]
                    / (2 * camera_config["Default"]["focal_length"])
                )
            json.dump(camera_config, file)
            fov = camera_config["Default"][fov_type]

        return fov


def camera_vector(h_angle: float, v_angle: float) -> Vector:
    """
    Generates a vector with an angle h_angle with the horizontal and an angle v_angle with the
    vertical.

    Using camera fovs will generate a vector that represents the corner of the camera's view.

    Parameters
    ----------
    h_angle : float
        The angle in radians to rotate horizontally
    v_angle : float
        The angle in radians to rotate vertically

    Returns
    -------
    camera_vector : Vector
        The vector which represents a given location in an image
    """
    # We know photos are being taken straight down, so the z axis is set to -1
    # After that we can use the tangent of the angle of each axis to get the resulting
    # vector pointing towards the pixel in the image

    vector: Vector = np.array([math.tan(-h_angle), math.tan(v_angle), -1], dtype=np.float64)
    vector = vector / np.linalg.norm(vector)

    vector = np.round(vector, decimals=6)

    return vector


def edge_angle(v_angle: float, h_angle: float) -> float:
    """
    Finds the angle in radians such that rotating by edge_angle on the Y axis then
    rotating by h_angle on the Z axis gives a vector an angle v_angle with the Y axis

    Can be derived using a square pyramid of height 1

    Parameters
    ----------
    v_angle : float
        The vertical angle
    h_angle : float
        The horizontal angle

    Returns
    -------
    edge_angle : float
        The angle in radians to rotate vertically
    """

    return np.arctan(np.tan(v_angle) * np.cos(h_angle))


def rotate_degrees(vector: Vector, rotation_deg: list[float]) -> Vector:
    """
    Rotates a vector based on a given roll, pitch, and yaw in degrees.

    Follows the MAVSDK.EulerAngle convention - positive roll is banking to the right, positive
    pitch is pitching nose up, positive yaw is clock-wise seen from above.

    Parameters
    ----------
    vector: Vector
        A vector represented by an XYZ coordinate that will be rotated
    rotation_deg: list[float]
        The [roll, pitch, yaw] in degrees to rotate

    Returns
    -------
    rotated_vector : Vector
        The vector which has been rotated
    """

    rotation_rad: list[float] = type_utils.assert_list_type(
        np.deg2rad(rotation_deg).tolist(),
        float,
    )

    return rotate_radians(vector, rotation_rad)


def rotate_radians(vector: Vector, rotation_rad: list[float]) -> Vector:
    """
    Rotates a vector based on a given roll, pitch, and yaw in radians.

    Follows the MAVSDK.EulerAngle convention - positive roll is banking to the right, positive
    pitch is pitching nose up, positive yaw is clock-wise seen from above.

    Parameters
    ----------
    vector: Vector
        A vector represented by an XYZ coordinate that will be rotated
    rotation_rad: list[float]
        The [roll, pitch, yaw] in radians to rotate

    Returns
    -------
    rotated_vector : Vector
        The vector which has been rotated
    """

    # Reverse the Y and Z rotation to match MAVSDK convention
    rotation_rad[0] *= -1
    rotation_rad[2] *= -1

    rotation = Rotation.from_euler("xyz", rotation_rad)
    result: Vector = rotation.apply(np.array(vector))

    return result
