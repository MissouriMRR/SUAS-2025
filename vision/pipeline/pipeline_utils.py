"""Pipeline functions not specific to either standard or emergent object"""

import json

import vision.common.constants as consts

from vision.common.bounding_box import BoundingBox

from vision.deskew.coordinate_lengths import get_coordinates
from vision.yolo.model import ObjectDetection

# We only care about the object classes that will actually appear in the competition
# You can see which are which here: https://github.com/WongKinYiu/yolov9/blob/main/data/coco.yaml
CLASS_PRIORITIES: dict[str, float] = {
    "person": 0.5,
    "car": 1.0,
    "motorcycle": 1.0,
    "airplane": 1.0,
    "bus": 1.0,
    "boat": 1.0,
    "stop sign": 1.0,
    "umbrella": 1.0,
    "suitcase": 1.0,
    "skis": 1.0,
    "snowboard": 1.0,
    "sports ball": 1.0,
    "baseball bat": 1.0,
    "tennis racket": 1.0,
    "bed": 1.0,
}


def read_parameter_json(json_path: str) -> dict[str, consts.CameraParameters]:
    """
    Will read in the data from the given json file and return it as a python dict.

    Parameters
    ----------
    json_path : str
        The path of a valid json file, assumed to have data in the same format as return type.

    Returns
    -------
    data : dict[str, CameraParameters]
        The python dict version of the data from the given json file.
    """

    with open(json_path, encoding="utf-8") as jfile:
        data: dict[str, consts.CameraParameters] = json.load(jfile)

    return data


def flyover_finished(state_path: str) -> bool:
    """
    Returns True if all photos have been taken and saved.
    The state_path file is a txt file containing only "True" if all images are taken

    Parameters
    ----------
    state_path: str
        The file holding a boolean

    Returns
    -------
    all_images_taken: bool
        True if all photos are saved
    """

    with open(state_path, encoding="UTF-8") as file:
        return str(file.read()) == "True"


def set_generic_attributes(
    box: BoundingBox,
    image_path: str,
    image_shape: consts.ImageShape,
    camera_parameters: consts.CameraParameters,
) -> bool:
    """
    Sets BoundingBox attributes by reference. Attributes changed are image_path, latitude,
    and longitude.

    "Generic" because these attributes are important for any object

    Parameters
    ----------
    box: BoundingBox
        The bounding box of the object to which the attributes will be set
    image_path: str
        The path for the image the bounding box is from
    image_shape : consts.ImageShape
        The shape of the image (returned by `image.shape` when image is a numpy image array)
    camera_parameters: CameraParameters
        The details of how and where the photo was taken

    Returns
    -------
    attributes_found: bool
        Returns true if all attributes were successfully found
    """

    box.set_attribute("image_path", image_path)

    coordinates: tuple[float, float] | None = get_coordinates(
        box.get_center_coord(), image_shape, camera_parameters
    )

    if coordinates is None:
        return False

    box.set_attribute("latitude", coordinates[0])
    box.set_attribute("longitude", coordinates[1])

    return True


def output_odlc_json(output_path: str, odlc_dict: consts.ODLCDict) -> None:
    """
    Saves the ODLC_Dict to a file

    Parameters
    ----------
    output_path: str
        The json file name and path to save the data in
    odlc_dict: consts.ODLC_Dict
        The dictionary of ODLCs matched with bottles
    """

    with open(output_path, "w", encoding="UTF-8") as file:
        json.dump(odlc_dict, file, indent=4)


def detection_to_bbox(
    detection: ObjectDetection, parameters: consts.CameraParameters
) -> BoundingBox:
    """
    Converts an ObjectDetection to a BoundingBox

    Parameters
    ----------
    detection: ObjectDetection
        The object detection to convert

    Returns
    -------
    bbox: BoundingBox
        The bounding box of the object detection
    """
    vertices = (
        (detection.bbox[0], detection.bbox[1]),
        (detection.bbox[2], detection.bbox[1]),
        (detection.bbox[2], detection.bbox[3]),
        (detection.bbox[0], detection.bbox[3]),
    )

    bbox = BoundingBox(vertices, detection.category)

    set_generic_attributes(bbox, detection.image, detection.shape, parameters)

    return bbox


def adjust_confidences(
    detections: list[ObjectDetection], expand_categories: bool = False, buffer: float = 0.0
) -> None:
    """
    Adjusts the confidence of ObjectDetections based on whether
    they are in the competition list or not.
    new_confidence = detection.confidence + buffer * CLASS_PRIORITIES[detection.category]

    Parameters
    ----------
    detections: list[ObjectDetection]
        The list of object detections to adjust
    expand_categories : bool, default False
        Whether to include categories not in the competition set
    buffer : float, default 0.0
        The value to add to confidence values of categories in the competition set
    """
    detection: ObjectDetection
    for detection in detections:
        if detection.category in CLASS_PRIORITIES:
            detection.confidence += buffer * CLASS_PRIORITIES[detection.category]
        elif not expand_categories:
            continue
