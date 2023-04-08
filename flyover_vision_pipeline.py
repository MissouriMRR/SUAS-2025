"""Runs the necessary Vision code during the flyover stage of competition"""

import cv2
import numpy as np

from typing import Callable
from vision.common.constants import Image, CameraParameters

from vision.competition_inputs.bottle_reader import load_bottle_info, BottleData
from vision.common.bounding_box import BoundingBox
from vision.common.odlc_characteristics import ODLCColor
from vision.emergent_object.emergent_object import create_emergent_model, detect_emergent_object

from vision.pipeline.standard_object import *
from vision.pipeline.emergent_object import *


def flyover_pipeline() -> None:
    """
    Finds all standard objects in each image in the input folder
    """

    # Load the data for each bottle
    bottle_info: list[BottleData] = load_bottle_info()

    # Load model
    emg_model: Callable[[Image], str] = create_emergent_model()
    
    # List of filenames for images already completed to prevent repeating work
    completed_images: list[str] = []
    
    # The list where all sightings of ODLCs will be stored
    saved_odlcs: list[BoundingBox] = []

    # The list of BoundingBoxes where all potential emergent objects will be stored
    saved_humanoids: list[BoundingBox] = []
    
    not_done: bool = True
    while not_done:
        image_parameters: dict[str, CameraParameters] = read_parameter_json()
        
        for image_path in image_parameters.keys:
            if image_path not in completed_images:
                image: Image = cv2.imread(image_path)
                
                camera_parameters: CameraParameters = image_parameters[image_path]
                
                saved_odlcs += find_standard_objects(image, camera_parameters, image_path)
                
                saved_humanoids += find_humanoids(image, emg_model, camera_parameters, image_path)
    
    # TODO:
    #   Pick emergent object
    #   Return airdrop locations






if __name__ == "__main__":
    original_image: Image = np.zeros((720, 1080, 3))
    image_path = "example_name.jpg"

    # These are made up, but shouldn't be *too* far off from realistic values
    camera_parameters: CameraParameters = {
        "focal_length": 10,
        "rotation_deg": [10, 20, 30],
        "drone_coordinates": [50.80085, 21.42069],
        "altitude_f": 100,
    }