"""
Bounding box objects represent an area in an image and
are used to convey information between flight and vision processes.
"""

from typing import Any, TypeAlias


import numpy as np

# A set of 4 coordinates that distinguish a region of an image.
# The order of the coordinates is (top-left, top-right, bottom-right, bottom-left).
Vertices: TypeAlias = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


def tlwh_to_vertices(tl_x: int, tl_y: int, width: int, height: int) -> Vertices:
    """
    Gets the vertices of a bounding box from a pixel, width, and height.

    Parameters
    ----------
    tl_x : int
        the top-left x pixels of the bounding box
    tl_y : int
        the top-left y pixel of the bounding box
    width : int
        the width of the bounding box
    height : int
        the height of the bounding box

    Returns
    -------
    vertices : Vertices
        Denotes the 4 pixels representing a box in an image.
        Vertices is a tuple of 4 pixels.
        Each pixel consists of a tuple 2 integers.
        Order is (top-left, top-right, bottom-right, bottom-left).
    """

    tl_coord: tuple[int, int] = (tl_x, tl_y)  # top left
    tr_coord: tuple[int, int] = (tl_x + width, tl_y)  # top right
    br_coord: tuple[int, int] = (tl_x + width, tl_y + height)  # bottom right
    bl_coord: tuple[int, int] = (tl_x, tl_y + height)  # bottom left

    return (tl_coord, tr_coord, br_coord, bl_coord)


class BoundingBox:
    """
    A set of 4 pixels that distinguish a region of an image.
    The top-left pixel, width, and height are used to create the
    BoundingBox.

    Attributes
    ----------
    center_lat_lon : tuple[float, float]
        the center latitude (center_lat_lon[0]) and longitude(center_lat_long[1]) of what the bounding box highlights

    Parameters
    ----------
    top_left : tuple[int, int]
        The top-left pixel of the BoundingBox.
    width : int
        The width of the BoundingBox.
    height : int
        The height of the BoundingBox.
    """

    center_lat_lon: tuple[float, float]

    def __init__(self, top_left: tuple[int, int], width: int, height: int) -> None:
        self.top_left: tuple[int, int] = top_left
        self.width: int = width
        self.height: int = height

        # Calculate the 4 vertices of the bounding box
        self.vertices: Vertices = tlwh_to_vertices(top_left[0], top_left[1], width, height)

    def get_x_vals(self) -> list[int]:
        """
        Gets the x values of the 4 pixels.

        Returns
        -------
        x_vals : list[int]
            The 4 x values of the vertices.
        """

        x_vals: list[int] = [vert[0] for vert in self.vertices]
        return x_vals

    def get_y_vals(self) -> list[int]:
        """
        Gets the y values of the 4 pixels.

        Returns
        -------
        y_vals : list[int]
            The 4 y values of the vertices.
        """

        y_vals: list[int] = [vert[1] for vert in self.vertices]
        return y_vals

    def get_x_extremes(self) -> tuple[int, int]:
        """
        Gets the minimum and maximum x values of the BoundingBox

        Returns
        -------
        min_x, max_x : tuple[int, int]
            The minimum and maximum x values.
        """

        x_vals: list[int] = self.get_x_vals()
        min_x: int = np.amin(x_vals)
        max_x: int = np.amax(x_vals)

        return min_x, max_x

    def get_y_extremes(self) -> tuple[int, int]:
        """
        Gets the minimum and maximum y values of the BoundingBox

        Returns
        -------
        min_y, max_y : tuple[int, int]
            The minimum and maximum y values.
        """

        y_vals: list[int] = self.get_y_vals()
        min_y: int = np.amin(y_vals)
        max_y: int = np.amax(y_vals)

        return min_y, max_y

    def get_x_avg(self) -> int:
        """
        Gets the average x pixel of the bounding box.

        Returns
        -------
        average : int
            the average of the 4 pixels' x-values
        """

        return int(np.mean(self.get_x_vals()))

    def get_y_avg(self) -> int:
        """
        Gets the average y pixel of the bounding box.

        Returns
        -------
        average : int
            the average of the 4 pixels' y-values
        """

        return int(np.mean(self.get_y_vals()))

    def get_center_coord(self) -> tuple[int, int]:
        """
        Gets the pixel of the center of the BoundingBox

        Returns
        -------
        center_pt : tuple[int, int]
            the pixel point at the center of the bounding box
        """

        return (self.get_x_avg(), self.get_y_avg())

    def get_width(self) -> int:
        """
        Get the width of the BoundingBox.

        Returns
        -------
        width: int
            the width of the BoundingBox based on max and min x values.
        """

        # Get the min and max x values, then calculate the width by subtracting
        # the min from the max
        min_x: int
        max_x: int
        min_x, max_x = self.get_x_extremes()
        width: int = max_x - min_x

        return width

    def get_height(self) -> int:
        """
        Get the height of the BoundingBox.

        Returns
        -------
        height: int
            the height of the BoundingBox based on max and min x values.
        """

        # Get the min and max y values, then calculate the height by
        # subtracting the min from the max
        min_y: int
        max_y: int
        min_y, max_y = self.get_y_extremes()
        height: int = max_y - min_y

        return height

    def get_width_height(self) -> tuple[int, int]:
        """
        Gets the width and height of the BoundingBox.

        Returns
        -------
        (width, height) : tuple[int, int]
            the width and height of the bounding box
        """

        return self.get_width(), self.get_height()


# Driver for testing functionality of BoundingBox object
if __name__ == "__main__":

    test_top_left: tuple[int, int] = (0, 0)
    test_width: int = 39
    test_height: int = 50

    # constructor
    object_bounds = BoundingBox(top_left=test_top_left, width=test_width, height=test_height)

    # width, height
    print("Width:", object_bounds.get_width())
    print("Height:", object_bounds.get_height())
    print("Width and Height:", object_bounds.get_width_height())

    # values, extremes, average
    print()
    print("X values:", object_bounds.get_x_vals())
    print("Y values:", object_bounds.get_y_vals())
    print("X extremes:", object_bounds.get_x_extremes())
    print("Y extremes:", object_bounds.get_y_extremes())
    print("X average:", object_bounds.get_x_avg())
    print("Y average:", object_bounds.get_y_avg())

    # center
    print()
    print("Center pixel:", object_bounds.get_center_coord())
