"""Distorts an image to generate an overhead view of flat terrain."""

from nptyping import NDArray, Shape, Float64
import cv2
import numpy as np
from vision.deskew.vector_utils import pixel_intersect
from vision.common.constants import Image, Corners, ImageInfo
from vision.deskew.coordinate_lengths import get_coordinates


def get_corner_points(image_shape: tuple[int, int, int] | tuple[int, int]) -> Corners:
    """
    Generates corner points of an image in the format
    1--2
    |  |
    4--3

    Parameters
    ----------
    image_shape: tuple[int, int, int] | tuple[int, int]
        Shape (num rows per axis) of the image

    Returns
    -------
    corner_points: Corners
        The corners of the image in clockwise order, starting at the top left
    """

    orig_height: int = image_shape[0]
    orig_width: int = image_shape[1]

    corner_points: Corners = np.array(
        [[0, 0], [orig_width, 0], [orig_width, orig_height], [0, orig_height]], dtype=np.float32
    )

    return corner_points


def get_relative_intersects(
    image_shape: tuple[int, int, int] | tuple[int, int],
    focal_length: float,
    rotation_deg: list[float],
    height: float = 1,
) -> Corners:
    """
    Calculates the pixel locations on the ground relative to the drone
    Units are same as `height`

    Parameters
    ----------
    image_shape: tuple[int, int, int] | tuple[int, int],
        The shape of the image to deskew. Aspect ratio should match the camera sensor
    focal_length : float
        The camera's focal length in millimeters - used to generate the camera's
        fields of view
    rotation_deg: list[float]
        The rotation of the drone in degrees. The constant ROTATION_OFFSET of the
        camera, stored in constants.py, will be applied first

    Returns
    -------
    intersects: Corners
        the location of the intersects relative to the drone
        X axis is north/south, Y axis is east/west

    """

    # Numpy converts `None` to NaN
    intersects: Corners = np.array(
        [
            pixel_intersect(point, image_shape, focal_length, rotation_deg, height)
            for point in get_corner_points(image_shape)
        ],
        dtype=np.float32,
    )

    return intersects


def perspective_matrix(
    image: ImageInfo,
    *,  # The following are keyword-only
    scale: float = 1,
) -> tuple[NDArray[Shape["3, 3"], Float64], Corners] | tuple[None, None]:
    """
    Generates a perspective transform matrix for deskewing an image

    Image is assumed to be the same aspect ratio as the drone camera.

    Returns (None, None) if the rotation and focal_length information does not generate a valid
    ending location.

    Parameters
    ----------
    image: ImageInfo
        The image with all of its information attached
    scale: float | None
        The scale of the projected image in unit of height
    interpolation: int | None
        The cv2 interpolation type to be used when deskewing.

    Returns
    -------
    (matrix, corner_points) : tuple[Image, Corners] | tuple[None, None]
        matrix : NDArray[Shape["3, 3"], Float64]
            The perspective transformation matrix for the image

            Returns None is no valid matrix could be generated

        dst_pts : Corners
            The destination corner points of the result in the image.
            Points are in order based on their location in the original image.
            Format is: (top left, top right, bottom right, bottom left), or
            1--2
            |  |
            4--3

            Returns None if no valid matrix could be generated.
    """

    orig_height: int = image["image_shape"]["height"]
    orig_width: int = image["image_shape"]["width"]

    # Generate points in the format
    # 1--2
    # |  |
    # 4--3
    source_pts: Corners = np.array(
        [[0, 0], [orig_width, 0], [orig_width, orig_height], [0, orig_height]], dtype=np.float32
    )

    # Numpy converts `None` to NaN
    intersects: Corners = np.array(
        [
            pixel_intersect(
                point,
                [orig_height, orig_width],
                image["camera_parameters"]["rotation_deg"],
                image["camera_parameters"]["altitude"],
            )
            for point in source_pts
        ],
        dtype=np.float32,
    )
    print(intersects)

    # Return (None, None) if any elements are NaN - camera vectors don't intersect the ground
    if np.any(np.isnan(intersects)):
        return None, None

    # Subtract the minimum on both axes so the minimum values on each axis are 0
    intersects -= np.min(intersects, axis=0)
    new_intersects = np.zeros(shape=(4, 2), dtype=np.float32)
    for i in range(4):
        for j in range(2):
            new_intersects[i][j] = intersects[i][j]

    # Scale the corner points to pixels per unit of height
    dst_pts: Corners = new_intersects * scale

    print("dst", dst_pts, "scale", scale, "\n src point", source_pts)
    # dst_pts[1],dst_pts[2]=dst_pts[2],dst_pts[1]
    matrix: NDArray[Shape["3, 3"], Float64] = cv2.getPerspectiveTransform(
        get_corner_points([orig_height, orig_width]), dst_pts
    )

    return matrix, dst_pts


def deskew(
    image: ImageInfo,
    *,  # The following are keyword-only
    scale: float = 1,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[Image, Corners] | tuple[None, None]:
    """
    Distorts an image to generate an overhead view of the photo. Parts of the image will be
    completely black where the camera could not see.

    Image is assumed to be a 3:2 aspect ratio to match the drone camera.

    Returns (None, None) if the rotation and focal_length information does not generate a valid
    ending location.

    Parameters
    ----------
    image : Image
        The input image to deskew along with all of its information. Aspect ratio should match the camera sensor
    scale: float | None
        Scales the resolution of the output. A value of 1 makes the area inside the camera view
        equal to the original image. Defaults to 1.
    interpolation: int | None
        The cv2 interpolation type to be used when deskewing.

    Returns
    -------
    (deskewed_image, corner_points) : tuple[Image, Corners] | tuple[None, None]
        deskewed_image : Image
            The deskewed image - the image is flattened with black areas in the margins

            Returns None if no valid image could be generated.

        dst_pts : Corners
            The corner points of the result in the image.
            Points are in order based on their location in the original image.
            Format is: (top left, top right, bottom right, bottom left), or
            1--2
            |  |
            4--3

            Returns None if no valid image could be generated.
    """

    image["center_coords"] = get_coordinates(
        (image["image_shape"]["width"] // 2, image["image_shape"]["height"] // 2),
        (image["image_shape"]["height"], image["image_shape"]["width"]),
        image["camera_parameters"],
    )
    print("Center coordinates:", image["center_coords"])

    matrix: NDArray[Shape["3, 3"], Float64]
    dst_pts: Corners
    matrix, dst_pts = perspective_matrix(image, scale=scale)

    if matrix is None or dst_pts is None:
        return None, None

    result_height: int = int(np.max(dst_pts[:, 1])) + 1
    result_width: int = int(np.max(dst_pts[:, 0])) + 1

    deskewed_image: Image = cv2.warpPerspective(
        image["image"],
        matrix,
        (result_width, result_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return deskewed_image, dst_pts.astype(np.int32)
