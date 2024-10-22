import unittest
import numpy as np
from vision.common.constants import CameraParameters, Point
from vision.common.bounding_box import BoundingBox
import vision.deskew as deskew
from vision.deskew.camera_distances import get_coordinates, bounding_area, calculate_distance


class TestVisionFunctions(unittest.TestCase):

    def setUp(self):
        # Set up the camera parameters and other constants for each test to avoid cross-contamination
        self.camera_params = CameraParameters(
            focal_length=35.0,
            rotation_deg=[0, 0, 0],
            drone_coordinates=[37.7749, -122.4194],
            altitude_f=1000.0,
        )
        self.image_shape = (1080, 1920, 3)  # Image size with 3 color channels

    def test_get_coordinates(self)-> None:
        # Test the coordinates calculation at the center of the image
        center_pixel = (960, 540)
        expected_coordinates = (37.77, -122.211)
        result = get_coordinates(center_pixel, self.image_shape, self.camera_params)
        self.assertIsNotNone(result, "Coordinates should not be None")
        self.assertAlmostEqual(result[0], expected_coordinates[0], places=2)
        self.assertAlmostEqual(result[1], expected_coordinates[1], places=2)

    def test_bounding_area(self)-> None:
        # Test bounding area calculation with a specific scenario
        self.camera_params.rotation_deg = [0, 0, 0]
        self.camera_params.drone_coordinates = [0, 0, 100]
        self.camera_params.altitude_f = 100.0

        self.box = BoundingBox(vertices=[(100, 200), (200, 200), (200, 300), (100, 300)])
        expected_area = 10000  # Example expected area for given parameters
        result = bounding_area(self.box, self.image_shape, self.camera_params)
        self.assertAlmostEqual(result, expected_area, msg="Bounding area calculation failed")

    def test_calculate_distance_invalid(self)-> None:
        # Test calculate distance with invalid pixel coordinates
        pixel1 = (100, 100)
        pixel2 = (-10, -10)  # Invalid pixel
        result = calculate_distance(pixel1, pixel2, self.image_shape, self.camera_params)
        self.assertIsNone(result, "Result should be None for invalid pixel coordinates")


if __name__ == "__main__":
    unittest.main()
