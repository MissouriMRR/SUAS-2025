"""
This file contains functions to filter contours to identify the odlc shapes from an already
processed image. The only function that should be needed is filter_contour(), the rest are helper
functions.
"""

import numpy as np
from nptyping import NDArray, Shape, UInt8, IntC, Float32, Bool8
import cv2
import vision.common.constants as consts
import vision.common.bounding_box as bbox


# constants
MIN_SHAPE_PIXEL_LEN: int = 30
# The minimum length/width (vertical/horizontal) a shape may have in pixels

BOX_AREA_RATIO_RANGE: float = 10.0
# The minimum acceptable ratio of image area to contour bounding box area in test_bounding_box()


def filter_contour(
    contour: consts.Contour,
    image_dims: tuple[int, int],
    approx_contour: consts.Contour,
) -> tuple[bool, bool]:
    """
    Runs all created test functions and handles logic to determine if a given contour (from the
    tuple of contours) is an ODLC shape or not

    Parameters
    ----------
    contour : consts.Contour
        A contour returned by the contour detection algorithm
    image_dims : tuple[int, int]
        The dimensions of the original entire image
        (only height and width as gotten from image.shape[:2], not the color channels)
        image_dim_height : int
            The height dimension of the image
        image_dim_width : int
            The width dimension of the image
    approx_contour : consts.Contour
        Parameter to provide the approximated version (from cv2.approxPolyDP) of the contour

    Returns
    -------
    shape_attrs : tuple[bool, bool, consts.Contour]
        is_odlc : bool
            True if the contour is (probably) an ODLC shape, False o.w.
        is_circular : bool
            True if is_odlc is true and the shape was detected to be circular
    """
    # calculate a bounding box of the contour to be used in multiple tests
    # tuple[int, int, int, int] is the return type of cv2.boundingRect()
    # the first two are the top left corner, last two are the width and height
    cnt_bound_box_retval: tuple[int, int, int, int] = cv2.boundingRect(contour)
    cnt_bound_box: bbox.BoundingBox = bbox.BoundingBox(
        bbox.tlwh_to_vertices(
            cnt_bound_box_retval[0],
            cnt_bound_box_retval[1],
            cnt_bound_box_retval[2],
            cnt_bound_box_retval[3],
        ),
        "object",
    )

    # test smallness, self intersect, bounding_box, and spikiness
    if (
        (not test_smallness(cnt_bound_box))
        or (not test_self_intersect(approx_contour))
        or (not test_bounding_box(cnt_bound_box, image_dims))
        or (not test_spikiness(contour))
    ):
        return (False, False)  # failed basic tests that should hold for all ODLC shapes
    return (True, False)  # passed ODLC tests so it is an object


def test_smallness(bounding_box: bbox.BoundingBox) -> bool:
    """
    Checks if the shape is less than MIN_SHAPE_PIXEL_LEN pixels vertically & horizontally.

    Parameters
    ----------
    contour : consts.Contour
        The individual contour to be evaluated (as returned from cv2.findContours)

    Returns
    -------
    is_big_enough : bool
        True if the shape is at least MIN_SHAPE_PIXEL_LEN long vertically and horizontally.
    """
    return (
        bounding_box.get_height() >= MIN_SHAPE_PIXEL_LEN
        and bounding_box.get_width() >= MIN_SHAPE_PIXEL_LEN
    )


def test_self_intersect(approx: consts.Contour) -> bool:
    """
    Checks if the given contour has self-intersections (its own lines cross eachother)

    Parameters
    ----------
    approx : consts.Contour
        The approximated version of the contour (from cv2.approxPolyDP())

    Returns
    -------
    no_self_intersect : bool
        True if the given contour does not have self-intersections.

    References
    ----------
    Opencv code documentation notes that a contour has self intersections if the list of indicies
    created with cv2.convexHull() is not monotonic. (See opencv github)
    """
    convex_hull: NDArray[Shape["*"], IntC] = np.squeeze(cv2.convexHull(approx, returnPoints=False))
    return bool(
        np.all(convex_hull[1:] >= convex_hull[:-1]) or np.all(convex_hull[0:-1] >= convex_hull[1:])
    )


def test_bounding_box(bounding_box: bbox.BoundingBox, dims: tuple[int, int]) -> bool:
    """
    Calculates the area of an upright bounding box around the given contour
    and compares it to the rectangle (image) of the given dimensions

    Parameters
    ----------
    contour : consts.Contour
        The individual contour to be evaluated (as returned from cv2.findContours)
    dims : tuple[int, int]
        The dimensions of the image the contour is from
        dim_height : int
            The height of the image
        dim_width : int
            The width of the image

    Returns
    -------
    acceptable_area_ratio : bool
        Returns true if the bounding box area is less than the image area by a factor of
        test_area_ratio or more
    """
    box_area: float = bounding_box.get_height() * bounding_box.get_width()
    img_area: float = dims[0] * dims[1]

    return box_area * BOX_AREA_RATIO_RANGE <= img_area


def test_spikiness(contour: consts.Contour) -> bool:
    """
    Checks whether the contour has some points that are much farther away from the center of mass
    than the rest of the points
    basically real shapes will not have random points that are really far away from the rest
    but a weird patch of grass or something without well defined edges will have crazy points
    all over the place

    Parameters
    ----------
    contour : consts.Contour
        The individual contour to be evaluated (as returned from cv2.findContours)

    Returns
    -------
    lacks_spikes : bool
        True unless there is a statistical outlier point that is uniquely far away (or close ig)

    References
    ----------
    This function uses a concept called the "Image moment" to calculate the center of a given
    contour. More information can be found here: https://en.wikipedia.org/wiki/Image_moment
    """
    moments: dict[str, float] = cv2.moments(contour)
    # com is Center of Mass, com = (x_coord, y_coord)
    # m10/m00 is the x coordinate of the center of the contour
    # m01/m00 is the y coordinate of the center of the contour
    com: NDArray[Shape["2"], Float32] = np.array(
        ((moments["m10"] / moments["m00"]), (moments["m01"] / moments["m00"])),
        dtype=np.float64,
    )

    # Holds the distance of each point in the contour to the center of the contour
    dists_com: NDArray[Shape["*"], Float32] = np.ndarray(contour.shape[0], Float32)
    for i in range(contour.shape[0]):
        dists_com[i] = cv2.norm(contour[i] - com)

    # calculate the average distance from the center of the contour
    dists_mean: float = np.mean(dists_com)
    # calculate the statistical outlier range of the set of distances to the center of all of the
    # points in the contour, this is 3*standard deviation of the set of distances
    dists_outlier_range: float = 3.0 * np.std(dists_com)

    # if all of the points are within 3 standard deviations of the avg distance to the center of
    # the contour, then there are no statistical outliers
    if np.all(dists_com < dists_mean + dists_outlier_range):
        return True
    return False


def min_common_bounding_box(contours: list[consts.Contour]) -> bbox.BoundingBox:
    """
    Takes a set of contours and returns the smallest bounding
    box that encloses all of them

    Parameters
    ----------
    contours : list[consts.Contour]
        Tuple of contours as formatted in cv2.findContour to find the minimum common bounding box

    Returns
    -------
    minimum_commom_bounding_box : bbox.BoundingBox
        The smallest bounding box that encompases all of the given contours
    """

    boxes: list[bbox.BoundingBox] = []
    for contour in contours:
        # tuple[int, int, int, int] is the return type of cv2.boundingRect()
        # the first two are the (y, x) of top left corner
        # the last two are the (y, x) of the bottom right corner
        contour_box: tuple[int, int, int, int] = cv2.boundingRect(contour)

        boxes.append(
            bbox.BoundingBox(
                bbox.tlwh_to_vertices(
                    contour_box[0],
                    contour_box[1],
                    contour_box[2],
                    contour_box[3],
                ),
                "object",
            )
        )

    min_x: int = np.min(np.array([box.get_x_extremes()[0] for box in boxes]))
    max_x: int = np.max(np.array([box.get_x_extremes()[1] for box in boxes]))
    min_y: int = np.min(np.array([box.get_y_extremes()[0] for box in boxes]))
    max_y: int = np.max(np.array([box.get_y_extremes()[1] for box in boxes]))
    min_box: bbox.BoundingBox = bbox.BoundingBox(
        ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)),
        "object",
    )
    return min_box


def generate_mask(contour: consts.Contour, box: bbox.BoundingBox) -> consts.Mask:
    """
    Will create a mask with the dimensions of the given bounding box that is true for any point
    that is inside the contour.

    Parameters
    ----------
    contour : consts.Contour
        The individual contour to be evaluated (as returned from cv2.findContours)
    box : bbox.BoundingBox
        The bounding box that will be used to create the mask image, the mask will be the size of
        box and will offset the contour so that it is inside of the box

    Returns
    -------
    contour_mask : consts.Mask
        A MaskType that is the dimensions of the input bounding box and is true wherever is
        inside of the input contour
    """
    dims: tuple[int, int] = box.get_width_height()[::-1]
    shifted_cnt: consts.Contour = contour - np.array([box.vertices[0][::-1]])

    mask: consts.Mask = np.zeros(dims)
    mask = cv2.drawContours(mask, [shifted_cnt], -1, True, cv2.FILLED).astype(Bool8)

    return mask


# All of main is some basic testing code
if __name__ == "__main__":
    # set to True and add points to raw_pts to test a polygon instead
    TESTING_POLYGON: bool = True

    # create a blank image
    test_image1: consts.Image = np.zeros([500, 500, 3], dtype=UInt8)

    # define some points to make a polygon, there is some draw circle function to try circles also
    raw_pts: NDArray[Shape["*, 2"], IntC] = np.array(
        [[249, 0], [499, 249], [249, 499], [0, 249]], IntC
    )
    pts: consts.Contour = raw_pts.reshape((-1, 1, 2))
    # put the points on the image
    if TESTING_POLYGON:
        test_image1 = cv2.polylines(test_image1, [pts], True, (255, 255, 255))
    else:
        test_image1 = cv2.circle(test_image1, (249, 249), 245, (255, 255, 255))

    cv2.imshow("pic", test_image1)
    cv2.waitKey(0)
    # make single channel to do contour stuff
    test_image: consts.ScImage = cv2.cvtColor(test_image1, cv2.COLOR_BGR2GRAY)

    # a variable length tuple is not good practice, but that is what opencv does here
    cnts_tmp: tuple[consts.Contour, ...]
    hier_tmp: consts.Hierarchy
    cnts_tmp, hier_tmp = cv2.findContours(test_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours1:", type(cnts_tmp), len(cnts_tmp), cnts_tmp)
    print("hierarchy1:", type(hier_tmp), hier_tmp.shape, hier_tmp)
    img2: consts.Image = np.dstack((test_image, test_image, test_image))

    # paint a filled in shape
    img2 = cv2.drawContours(img2, cnts_tmp, 0, (255, 255, 255), thickness=cv2.FILLED)
    cv2.imshow("cnts1-0", img2)
    cv2.waitKey(0)

    # find the contours for "real." My code doesnt do that, this is just to generate test contours
    cnts: tuple[consts.Contour, ...]
    hier: consts.Hierarchy
    cnts, hier = cv2.findContours(test_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours2:", type(cnts), len(cnts), cnts)
    print("hierarchy2:", type(hier), hier.shape, hier)

    # for each contour run all of the tests on it
    ind: int
    cntr: consts.Contour
    for ind, cntr in enumerate(cnts):
        # the next 4 "lines" of code create a cropped ScImage around the contour to pass to
        # circle detection function
        cntr_bbox_retval: tuple[int, int, int, int] = cv2.boundingRect(cntr)
        cntr_bbox: bbox.BoundingBox = bbox.BoundingBox(
            bbox.tlwh_to_vertices(
                cntr_bbox_retval[0],
                cntr_bbox_retval[1],
                cntr_bbox_retval[2],
                cntr_bbox_retval[3],
            ),
            "object",
        )
        cntr_msk: consts.Mask = generate_mask(cntr, cntr_bbox)
        cntr_sc_img: consts.ScImage = np.where(cntr_msk, 255, 0).astype(np.uint8)
        cv2.imshow(f"cntr{ind}_sc_img", np.dstack((cntr_sc_img, cntr_sc_img, cntr_sc_img)))
        cv2.waitKey(0)
        print(type(cntr_sc_img), type(cntr_sc_img[0]), type(cntr_sc_img[0, 0]))
        # actually running each test individually and printing results for testing/debugging
        print("\nSmallness Test:", test_smallness(cntr))
        print(
            "Bounding Box Test:",
            test_bounding_box(cntr, (test_image.shape[0], test_image.shape[1])),
        )
        print("Jaggedness Test:", test_spikiness(cntr))
