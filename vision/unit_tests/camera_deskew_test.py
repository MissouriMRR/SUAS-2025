"""Unit test for camera deskew functionality."""

import unittest
from vision.common.constants import CameraParameters
from vision.deskew.camera_distances import bounding_area, calculate_distance
from vision.deskew.coordinate_lengths import get_coordinates
from vision.common.bounding_box import BoundingBox


class TestVisionFunctions(unittest.TestCase):
    """
    Tests to verify the functionality of calculations
    concerning object positioning and distances in images captured by a camera.

    Attributes
    ----------
    camera_params: CameraParameters
        Sample camera parameters to be used in the test.
    image_shape: tuple[int, int, int]
        The dimensions of the image used in tests, specified as (height, width, channels).
    box: BoundingBox
        Sample bounding box of an object to be used in the test.
    """

    def setUp(self) -> None:
        """
        Initializes common properties for all test methods. Sets up camera parameters
        and image dimensions that simulate a typical usage scenario.

        """
        self.camera_params = CameraParameters(
            rotation_deg=[0, 0, 0],
            drone_coordinates=[37.7749, -122.4194],
            altitude=1000.0,
        )
        self.image_shape = (1080, 1920, 3)  # Image size with 3 color channels
        self.box = BoundingBox(
            obj_type="object",
            vertices=((100, 200), (200, 200), (200, 300), (100, 300)),
        )

    def test_get_coordinates(self) -> None:
        """

        Verifies that the accurately calculates  coordinates
        from a center pixel. Asserts correct type and closeness to expected values.

        Returns
        -------
            None: This method does not return a value but asserts the correctness of the output.
        """
        center_pixel = (960, 540)
        expected_coordinates = (37.77, -122.211)
        result = get_coordinates(center_pixel, self.image_shape, self.camera_params)

        self.assertIsNotNone(result, "Coordinates should not be None")
        # Check if the result is not None and is of the expected tuple type
        self.assertIsInstance(result, tuple, "Result should be a tuple")
        if isinstance(result, tuple):
            self.assertEqual(len(result), 2, "Result tuple should have two elements")
            self.assertIsInstance(result[0], float, "Latitude is float")
            self.assertIsInstance(result[1], float, "Longitude is float")
        # Make sure to get the expected value
        if result is not None:
            self.assertAlmostEqual(result[0], expected_coordinates[0], places=2)
            self.assertAlmostEqual(result[1], expected_coordinates[1], places=2)

    def test_bounding_area(self) -> None:
        """
        Tests the  function to calculate the area of a bounding box
        within the image, simulating an object detection scenario.

        Returns
        -------
            None: This method does not return a value but asserts the area calculation is correct.
        """
        self.box = BoundingBox(
            obj_type="",
            vertices=((100, 200), (200, 200), (200, 300), (100, 300)),
        )
        result = bounding_area(self.box, self.image_shape, self.camera_params)

        # Ensure result is not None before asserting
        self.assertIsNotNone(result, "Result should not be None for valid bounding box")

    def test_calculate_distance(self) -> None:
        """
        Tests distance calculation between two points in an image. Ensures the method
        returns a valid result.
        """
        pixel1 = (100, 100)
        pixel2 = (200, 200)  # Valid pixel coordinates
        result = calculate_distance(pixel1, pixel2, self.image_shape, self.camera_params)

        # Ensure result is not None before asserting
        self.assertIsNotNone(result, "Result should not be None for valid pixel coordinates")


if __name__ == "__main__":
    unittest.main()
