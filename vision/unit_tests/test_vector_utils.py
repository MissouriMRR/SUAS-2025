"""
Testing vision.deskew.vector_utils.py
"""

import json
import unittest
import numpy as np
from utils import type_utils
from scipy.spatial.transform import Rotation

import vision.deskew.vector_utils as vu
from vision.common.constants import Vector, Point, ROTATION_OFFSET, CameraConfig


Test_IHAT: Vector = np.array([1, 0, 0], dtype=np.float64)

# NOTE: These tests are ordered with the helper functions first before the
#       other functions that use them


class TestIHAT(unittest.TestCase):
    """
    Testing the IHAT constant from the vector utils module
    """

    def test_ihat_type(self) -> None:
        """
        Asserts that IHAT is of type Vector

        Raises
        ------
        AssertionError
            If IHAT is not of type Vector
        """

        # Fail message
        fail_msg: str = "IHAT is not of type Vector"

        self.assertEqual(type(vu.IHAT), type(Test_IHAT), fail_msg)

    def test_ihat_value(self) -> None:
        """
        Asserts that IHAT has the value of np.array([1, 0, 0])

        Raises
        ------
        AssertionError
            If IHAT does not have the value of np.array([1, 0, 0])
        """

        # This doesn't need a fail message as we will use the one that numpy
        # provides

        # Test the value of IHAT using numpy's assert_array_equal function
        try:
            np.testing.assert_array_equal(Test_IHAT, vu.IHAT)
        except AssertionError as error_msg:
            self.fail(error_msg)


class TestPixelAngle(unittest.TestCase):
    """
    Testing the pixel_angle function from the vector utils module
    """

    def test_pixel_angle(self) -> None:
        """
        Asserts that pixel_angle returns the expected value

        Raises
        ------
        AssertionError
            If pixel_angle returns a value other than what was calculated
        """

        # Test values
        test_fov: float = 90.6
        test_ratio: float = 0.17

        # Expected value
        expected: float = np.arctan(np.tan(test_fov / 2) * (1 - 2 * test_ratio))

        # Test the function
        result: float = vu.pixel_angle(test_fov, test_ratio)

        # Fail message
        fail_msg: str = "pixel_angle returned an incorrect value"

        self.assertEqual(expected, result, fail_msg)


class TestPlaneCollision(unittest.TestCase):
    """
    Testing the plane_collision function from the vector utils module
    """

    def test_plane_collision(self) -> None:
        """
        Test plane_collision with a known example

        Raises
        ------
        AssertionError
            If plane_collision returns a value other than what was calculated
        """

        # Test values
        test_ray_direction: Vector = np.array([5, 4, 9], dtype=np.float64)
        test_height: float = 63.5

        # Calculate the time
        test_time: float = -test_height / test_ray_direction[2].item()

        # Create a value for the intersection
        test_intersect: Point | None

        # Check if the time is valid
        if np.isinf(test_time) or np.isnan(test_time) or test_time < 0:
            test_intersect = None
        else:
            test_intersect = test_ray_direction[:2] * test_time

        # Expected value
        expected: Point | None = test_intersect

        # Test the function
        result: Point | None = vu.plane_collision(test_ray_direction, test_height)

        # This only needs a fail message if the expected value is not None
        fail_msg: str = "plane_collision returned an incorrect value"

        if expected is None:
            self.assertIsNone(result, fail_msg)
            return

        # This is required, otherwise the mypy linter will complain about
        # incompatible types in the assert_array_equal function
        result_point: Point = result

        # Test using numpy's assert_array_equal function
        try:
            np.testing.assert_array_equal(expected, result_point)
        except AssertionError as error_msg:
            # Use the fail message from numpy
            self.fail(error_msg)


class TestGetFOV(unittest.TestCase):
    """
    Testing the get_fov function from the vector utils module
    """

    def test_get_fov(self) -> None:
        """
        Test get_fov with a known example

        Raises
        ------
        AssertionError
            If get_fov returns a value other than what was calculated
        """

        with open("vision/common/camera_config.json", "rw", encoding="ascii") as file:
            camera_config: CameraConfig = json.load(file)
            test_config: CameraConfig = {
                "Default": {
                    "sensorHeight": 22,
                    "sensorWidth": 22,
                    "horizontalFOV": 0,
                    "verticalFOV": 0,
                    "focal_length": 45.5,
                },
                "Airsim": {
                    "sensorHeight": 0,
                    "sensorWidth": 0,
                    "horizontalFOV": 0,
                    "verticalFOV": 0,
                    "focal_length": 0,
                },
                "airsim_flag": False,
            }
            json.dump(test_config, file)

            # Test values
            test_focal_length: float = 45.5
            test_sensor_size: float = 22

            # Expected value
            expected: float = 2 * np.arctan(test_sensor_size / (2 * test_focal_length))

            # Test the function
            result: tuple[float, float] = vu.get_fov()

            # Fail message
            fail_msg: str = "get_fov returned an incorrect value"

            self.assertEqual(expected, result[0], fail_msg)
            json.dump(camera_config, file)


class TestEdgeAngle(unittest.TestCase):
    """
    Testing the edge_angle function from the vector utils module
    """

    def test_edge_angle(self) -> None:
        """
        Test edge_angle with a known example

        Raises
        ------
        AssertionError
            If edge_angle returns a value other than what was calculated
        """

        # Test values
        test_vertical_angle: float = 0.5
        test_horizontal_angle: float = 0.3

        # Expected value
        expected: float = np.arctan(np.tan(test_vertical_angle) * np.cos(test_horizontal_angle))

        # Test the function
        result: float = vu.edge_angle(test_vertical_angle, test_horizontal_angle)

        # Fail message
        fail_msg: str = "edge_angle returned an incorrect value"

        self.assertEqual(expected, result, fail_msg)


class TestRotateRadians(unittest.TestCase):
    """
    Testing the rotate_radians function from the vector utils module
    """

    def test_rotate_radians(self) -> None:
        """
        Test rotate_radians with a known example

        Raises
        ------
        AssertionError
            If rotate_radians returns a value other than what was calculated
        """

        # Test values
        test_vector: Vector = np.array([5, 3, 48], dtype=np.float64)
        test_rotation_radians: list[float] = [0.5, 0.3, 0.2]
        # Make a copy of the rotation degrees because of Python's mutable lists
        test_rotation_radians_func: list[float] = test_rotation_radians.copy()

        # Do the math for the
        # Flip the Y and Z rotation
        test_rotation_radians[1] *= -1
        test_rotation_radians[2] *= -1

        # Expected value
        expected: Vector = Rotation.from_euler("xyz", test_rotation_radians).apply(
            np.array(test_vector)
        )

        # Test the function
        result: Vector = vu.rotate_radians(test_vector, test_rotation_radians_func)

        # This doesn't need a fail message as we will use the one that numpy
        # provides

        # Test using numpy's assert_array_equal function
        try:
            np.testing.assert_array_equal(expected, result)
        except AssertionError as error_msg:
            self.fail(error_msg)


class TestRotateDegrees(unittest.TestCase):
    """
    Testing the rotate_degrees function from the vector utils module
    """

    def test_rotate_degrees(self) -> None:
        """
        Test rotate_degrees with a known example

        Raises
        ------
        AssertionError
            If rotate_degrees returns a value other than what was calculated
        """

        # Test values
        test_vector: Vector = np.array([1, 0, 0], dtype=np.float64)
        test_rotation_degrees: list[float] = [4.8, 9.1, 43.2]

        # Convert degrees to radians
        test_rotation_radians: list[float] = type_utils.assert_list_type(
            np.deg2rad(test_rotation_degrees).tolist(),
            float,
        )

        # Expected value
        expected: Vector = vu.rotate_radians(
            test_vector, test_rotation_radians
        )  # This function has been tested

        # Test the function
        result: Vector = vu.rotate_degrees(test_vector, test_rotation_degrees)

        # This doesn't need a fail message as we will use the one that numpy
        # provides

        # Test using numpy's assert_array_equal function
        try:
            np.testing.assert_array_equal(expected, result)
        except AssertionError as error_msg:
            self.fail(error_msg)


class TestCameraVector(unittest.TestCase):
    """
    Testing the camera_vector function from the vector utils module
    """

    def test_camera_vector(self) -> None:
        """
        Test camera_vector with a known example

        Raises
        ------
        AssertionError
            If camera_vector returns a value other than what was calculated
        """

        # Test values
        test_horizontal_angle: float = 4.3
        test_vertical_angle: float = 9.5

        # Calculate the edge angle
        test_edge: float = vu.edge_angle(
            test_vertical_angle, test_horizontal_angle
        )  # This function has been tested

        # Expected value
        expected: Vector = vu.rotate_radians(
            Test_IHAT, [0, test_edge, -test_horizontal_angle]
        )  # This function has been tested

        # Test the function
        result: Vector = vu.camera_vector(test_horizontal_angle, test_vertical_angle)

        # This doesn't need a fail message as we will use the one that numpy
        # provides

        # Test using numpy's assert_array_equal function
        try:
            np.testing.assert_array_equal(expected, result)
        except AssertionError as error_msg:
            self.fail(error_msg)


class TestPixelVector(unittest.TestCase):
    """
    Testing the pixel_vector function from the vector utils module
    """

    def test_pixel_vector(self) -> None:
        """
        Test pixel_vector with a known example

        Raises
        ------
        AssertionError
            If pixel_vector returns a value other than what was calculated
        """

        # Test values
        test_pixel: tuple[int, int] = (35, 89)
        test_image_shape: tuple[int, int] = (100, 200)

        # Calculate the FOVs
        test_horizontal_fov: float
        test_vertical_fov: float
        test_horizontal_fov, test_vertical_fov = vu.get_fov()  # This function has been tested

        # Expected value
        expected: Vector = vu.camera_vector(
            vu.pixel_angle(test_horizontal_fov, test_pixel[0] / test_image_shape[1]),
            vu.pixel_angle(test_vertical_fov, test_pixel[1] / test_image_shape[0]),
        )  # These functions have been tested

        # Test the function
        result: Vector = vu.pixel_vector(test_pixel, test_image_shape)

        # This doesn't need a fail message as we will use the one that numpy
        # provides

        # Test using numpy's assert_array_equal function
        try:
            np.testing.assert_array_equal(expected, result)
        except AssertionError as error_msg:
            self.fail(error_msg)


class TestPixelIntersect(unittest.TestCase):
    """
    Testing the pixel_intersect function from the vector utils module
    """

    def test_pixel_intersect(self) -> None:
        """
        Test pixel_intersect with a known example

        Raises
        ------
        AssertionError
            If pixel_intersect returns a value other than what was calculated
        """

        # Test values
        test_pixel: tuple[int, int] = (93, 72)
        test_image_shape: tuple[int, int] = (300, 600)
        test_height: float = 33.3
        test_rotation_degrees: list[float] = [4.5, 0.3, 8.2]
        # Make a copy of the rotation degrees because of Python's mutable lists
        test_rotation_degrees_func: list[float] = test_rotation_degrees.copy()

        # Calculate the pixel vector
        test_vector: Vector = vu.pixel_vector(test_pixel, test_image_shape)
        # This function above has been tested

        # Calculate the rotated vector
        test_vector_rotated: Vector = vu.rotate_degrees(test_vector, ROTATION_OFFSET)
        # This function above has been tested

        # Calculate the drone rotation
        test_vector_drone_rotated: Vector = vu.rotate_degrees(
            test_vector_rotated, test_rotation_degrees
        )

        # Calculate the intersection
        test_intersect: Point | None = vu.plane_collision(test_vector_drone_rotated, test_height)
        # This function above has been tested

        # Expected value
        expected: Point | None = test_intersect

        # Test the function
        result: Point | None = vu.pixel_intersect(
            test_pixel, test_image_shape, test_rotation_degrees_func, test_height
        )

        # This only needs a fail message if the expected value is not None
        fail_msg: str = "pixel_intersect returned an incorrect value"

        if expected is None:
            self.assertIsNone(result, fail_msg)
            return

        # This is required, otherwise the mypy linter will complain about
        # incompatible types in the assert_array_equal function
        result_point: Point = result

        # Test using numpy's assert_array_equal function
        try:
            np.testing.assert_array_equal(expected, result_point)
        except AssertionError as error_msg:
            # Use the fail message from numpy
            self.fail(error_msg)


if __name__ == "__main__":
    unittest.main()
