"""Distorts an image to generate an overhead view of flat terrain."""

from nptyping import NDArray, Shape, Float64

import cv2
import numpy as np

from vision.deskew.vector_utils import pixel_intersect
from vision.common.constants import Image, Corners


def perspective_matrix(
    image_shape: tuple[int, int, int] | tuple[int, int],
    focal_length: float,
    rotation_deg: list[float],
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
    image_shape: tuple[int, int, int] | tuple[int, int],
        The shape of the image to deskew. Aspect ratio should match the camera sensor
    focal_length : float
        The camera's focal length in millimeters - used to generate the camera's
        fields of view
    rotation_deg: list[float]
        The rotation of the drone in degrees. The constant ROTATION_OFFSET of the
        camera, stored in constants.py, will be applied first
    scale: float | None
        Scales the resolution of the output. A value of 1 makes the area inside the camera view
        equal to the original image. Defaults to 1.
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

    orig_height: int = image_shape[0]
    orig_width: int = image_shape[1]

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
            pixel_intersect(point, image_shape, focal_length, rotation_deg, 1)
            for point in source_pts
        ],
        dtype=np.float32,
    )

    # Return (None, None) if any elements are NaN - camera vectors don't intersect the ground
    if np.any(np.isnan(intersects)):
        return None, None

    # Flip the endpoints over the X axis (top left is 0,0 for images)
    intersects[:, 1] *= -1

    # Subtract the minimum on both axes so the minimum values on each axis are 0
    intersects -= np.min(intersects, axis=0)

    # Find the area using cv2 contour tools
    area: float = cv2.contourArea(intersects)

    # Scale the output so the area of the important pixels is about the same as the starting image
    target_area: float = orig_height * orig_width * scale
    intersect_scale: np.float64 = np.float64(np.sqrt(target_area / area))
    dst_pts: Corners = intersects * intersect_scale

    matrix: NDArray[Shape["3, 3"], Float64] = cv2.getPerspectiveTransform(source_pts, dst_pts)

    return matrix, dst_pts


def deskew(
    image: Image,
    focal_length: float,
    rotation_deg: list[float],
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
        The input image to deskew. Aspect ratio should match the camera sensor
    focal_length : float
        The camera's focal length in millimeters - used to generate the camera's
        fields of view
    rotation_deg: list[float]
        The rotation of the drone in degrees. The constant ROTATION_OFFSET of the
        camera, stored in constants.py, will be applied first
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

    matrix: NDArray[Shape["3, 3"], Float64]
    dst_pts: Corners
    matrix, dst_pts = perspective_matrix(image.shape, focal_length, rotation_deg, scale=scale)

    if matrix is None or dst_pts is None:
        return None, None

    result_height: int = int(np.max(dst_pts[:, 1])) + 1
    result_width: int = int(np.max(dst_pts[:, 0])) + 1

    deskewed_image: Image = cv2.warpPerspective(
        image,
        matrix,
        (result_width, result_height),
        flags=interpolation,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    return deskewed_image, dst_pts.astype(np.int32)
