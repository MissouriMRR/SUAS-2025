"""Functions that use vectors to calculate camera intersections with the ground"""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from utils import type_utils
from vision.common.constants import Point, Vector, CameraConfig


# Vector pointing toward the +X axis, represents the camera's forward direction when the
#   rotation on all axes is 0
IHAT: Vector = np.array([1, 0, 0], dtype=np.float64)


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
        The [roll, pitch, yaw] rotation of the drone in degrees
    height : float
        The height that the image was taken at. The units of the output will be the units of the
        input.

    Returns
    -------
    intersect : Point | None
        The coordinates [X,Y] where the pixel's vector intersects with the ground. Units
            are the same as `height`
        Returns None if there is no intersect.
    """

    # Create the normalized vector representing the direction of the given pixel
    vector: Vector = pixel_vector(pixel, image_shape)

    # Apply the drone rotation
    vector = rotate_degrees(vector, rotation_deg)

    intersect: Point | None = plane_collision(vector, height)

    return intersect


def plane_collision(ray_direction: Vector, height: float) -> Point | None:
    """
    Returns the point where a ray intersects the XY plane. North is +X
    Returns None if there is no intersect.

    Parameters
    ----------
    ray_direction : Vector
        XYZ coordinates that represent the direction a ray faces
    height : float
        The Z coordinate for the starting height of the ray; can be any units

    Returns
    -------
    intersect : Point | None
        The ray's intersection with the plane in [X,Y] format. Units are the same as
        `height`
        Returns None if there is no intersect.
    """

    # Find the "time" at which the line intersects the plane.
    # Line is defined as ray_direction * time + vertex. Vertex is the point at
    #   X, Y, Z = (0, 0, height)
    time: float = -height / ray_direction[2].item()

    # Checks if the ray intersects with the plane - negative `time` means the intersection
    #   is behind the camera
    if np.isinf(time) or np.isnan(time) or time < 0:
        return None

    intersect: Point = ray_direction[:2] * time

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
        The pixel's angle from the center of the camera along a single axis
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

            if camera_config["Airsim"]["horizontalFOV"] != 0:
                v_fov = camera_config["Airsim"]["verticalFOV"]
            else:
                v_fov = calculate_fov("Airsim", "verticalFOV")
        else:
            if camera_config["Default"]["horizontalFOV"] != 0:
                h_fov = camera_config["Default"]["horizontalFOV"]
            else:
                h_fov = calculate_fov("Default", "horizontalFOV")

            if camera_config["Default"]["horizontalFOV"] != 0:
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
    with open("vision/common/camera_config.json", "rw", encoding="ascii") as file:
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

    # Calculate the vertical rotation needed for the final vector to have the desired direction
    edge: float = edge_angle(v_angle, h_angle)

    vector: Vector = rotate_radians(IHAT, [0, edge, -h_angle])

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
    rotation_rad[1] *= -1
    rotation_rad[2] *= -1

    result: Vector = Rotation.from_euler("xyz", rotation_rad).apply(np.array(vector))

    return result
