"""Functions that perform standard object detection, localization, and classification"""

from typing import Iterable, TypeAlias

import utm

from flight.extract_gps import extract_gps, GPSData
from flight.waypoint.calculate_distance import calculate_distance
from flight.waypoint.geometry import Point

from state_machine.flight_settings import FlightSettings

import vision.common.constants as consts

from vision.common.bounding_box import BoundingBox

from vision.standard_object.odlc_contour_detection import fetch_shape_contours
from vision.standard_object.odlc_classify_shape import process_shapes

import vision.pipeline.pipeline_utils as pipe_utils
from vision.yolo.model import ObjectDetection

ContourHierarchyList: TypeAlias = list[tuple[tuple[consts.Contour, ...], consts.Hierarchy]]


def find_standard_objects(
    original_image: consts.Image,
    camera_parameters: consts.CameraParameters,
    image_path: str,
) -> list[BoundingBox]:
    """
    Finds all bounding boxes of standard objects in an image

    Parameters
    ----------
    original_image: Image
        The image to find shapes in
    camera_parameters: CameraParameters
        The details of how and where the photo was taken
    image_path: str
        The path for the image the bounding box is from

    Returns
    -------
    found_odlcs: list[BoundingBox]
        The list of bounding boxes of detected standard objects
    """

    found_odlcs: list[BoundingBox] = []
    contours: list[consts.Contour] = fetch_shape_contours(original_image, True, "contours.jpg")
    shapes: list[BoundingBox] = process_shapes(contours)
    shape: BoundingBox
    for shape in shapes:
        # Set the shape attributes by reference. If successful, keep the shape
        if pipe_utils.set_generic_attributes(shape, original_image.shape, camera_parameters):
            found_odlcs.append(shape)

    return found_odlcs


def create_odlc_dict(
    bounding_boxes: Iterable[BoundingBox], flight_settings: FlightSettings
) -> consts.ODLCDict:
    """
    Creates the ODLCDict dictionary from a list of shape bounding boxes.
    Discards bounding boxes whose center is not inside the airdrop boundary.

    Parameters
    ----------
    bounding_boxes : Iterable[BoundingBox]
        An iterable of the sightings of each object.
    flight_settings : FlightSettings
        The flight settings.
        Used to get the airdrop boundary.
    Returns
    -------
    odlc_dict : consts.ODLCDict
        The dictionary of ODLCs matching the output format.
    """

    # Get ODLC boundary
    gps_data: GPSData = extract_gps(flight_settings.mission_data_path)
    odlc_boundary: list[Point] = [
        Point(odlc_boundary_point.easting, odlc_boundary_point.northing)
        for odlc_boundary_point in gps_data["odlc_boundary_utm"]
    ]
    zone_number: int = gps_data["odlc_boundary_utm"][0].zone_number
    zone_letter: str = gps_data["odlc_boundary_utm"][0].zone_letter

    odlc_dict: consts.ODLCDict = {}

    bbox: BoundingBox
    for index, bbox in enumerate(bounding_boxes):
        # Check if in bounds
        easting: float
        northing: float
        easting, northing, _, _ = utm.from_latlon(
            bbox.center_lat_lon[0],
            bbox.center_lat_lon[1],
            force_zone_number=zone_number,
            force_zone_letter=zone_letter,
        )
        if not Point(easting, northing).is_inside_shape(odlc_boundary):
            continue

        odlc_dict[str(index)] = {
            "latitude": bbox.center_lat_lon[0],
            "longitude": bbox.center_lat_lon[1],
        }
    return odlc_dict


def proximity_check(
    detections: list[ObjectDetection],
    parameters: dict[str, consts.CameraParameters],
    min_distance: float = 7.0,
) -> list[tuple[ObjectDetection, BoundingBox]]:
    """
    Checks for detections that are too close to each other, and
    removes the one with lower confidence.
    The ruleset states that objects must be at least 50 feet apart.

    Parameters
    ----------
    detections : list[ObjectDetection]
        A list with all object detections.
    parameters : dict[str, consts.CameraParameters]
        A dictionary with the image parameters for every
        captured image.
    min_distance : float, optional
        The minimum distance between objects in METERS, by default 7.0

    Returns
    -------
    filtered_detections : list[tuple[ObjectDetection, BoundingBox]]
        All detections that are at least `min_distance` apart, along with
        their bounding boxes.
    """
    filtered_detections: list[tuple[ObjectDetection, BoundingBox]] = []
    # If we sort the detections by confidence, we can stop as soon as we find a collision
    detections.sort(key=lambda entry: entry.confidence, reverse=True)

    for detection in detections:
        image_name: str = detection.image.split("/")[-1]
        image_parameters: consts.CameraParameters = parameters[image_name]
        bounding_box: BoundingBox = pipe_utils.detection_to_bbox(detection, image_parameters)
        latitude: float = bounding_box.get_attribute("latitude")
        longitude: float = bounding_box.get_attribute("longitude")

        collided: bool = False
        existing_bbox: BoundingBox
        for _, existing_bbox in filtered_detections:
            existing_latitude: float = existing_bbox.get_attribute("latitude")
            existing_longitude: float = existing_bbox.get_attribute("longitude")

            # We can use 0 altitude for calculation as the search area is pretty flat
            distance: float = calculate_distance(
                latitude, longitude, 0, existing_latitude, existing_longitude, 0
            )
            if distance < min_distance:
                collided = True
                break

        if not collided:
            filtered_detections.append((detection, bounding_box))

    return filtered_detections
