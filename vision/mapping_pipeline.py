"""Runs the necessary code for the Mapping component of the competition"""

import logging
from ctypes import c_bool
from multiprocessing.sharedctypes import SynchronizedBase  # pylint: disable=unused-import

import asyncio
import cv2
import numpy as np

from vision.mapping.map import Map

import vision.common.constants as consts
import vision.pipeline.pipeline_utils as pipe_utils
from vision.deskew.camera_distances import pixel_per_foot


# Disable duplicate code checking because the flyover pipeline is similar
# pylint: disable=duplicate-code
async def mapping_pipeline(
    camera_data_path: str, image_dir: str, state_path: str, output_path: str
) -> None:
    """
    Runs the code that generates the map from a folder of photos

    Parameters
    ----------
    camera_data_path: str
        The path to the json file containing the CameraParameters entries
    state_path: str
        A text file containing True if all images have been taken and False otherwise
    output_path: str
        The json file name and path to save the data in
    """

    # List of filenames for images already completed to prevent repeating work
    completed_images: list[str] = []

    image_parameters: dict[str, consts.CameraParameters] = pipe_utils.read_parameter_json(
        camera_data_path
    )
    while len(image_parameters) == 0:
        await asyncio.sleep(1)
        image_parameters: dict[str, consts.CameraParameters] = pipe_utils.read_parameter_json(
            camera_data_path
        )
    # using first image taken for mapping to determine pixels per foot
    first_image: str = next(iter(image_parameters))

    map: Map = Map(
        pixel_per_foot(
            cv2.imread(image_dir + "/" + first_image).shape, image_parameters[first_image]
        )
    )

    # Wait for and process unfinished images until no more images are being taken
    all_images_taken: c_bool = c_bool(True)
    first_check = True

    while not all_images_taken or first_check:
        # Wait to check the file instead of spamming it
        if not first_check:
            await asyncio.sleep(1)

        first_check = False

        # Check if all images have been taken
        # all_images_taken = capture_status.value  # type: ignore

        # Load in the json containing the camera data
        image_parameters: dict[str, consts.CameraParameters] = pipe_utils.read_parameter_json(
            camera_data_path
        )

        # Loop through all images in the json - if it hasn't been processed, process it
        for image_path in image_parameters.keys():
            # image_path = image_dir + image_path

            if image_path not in completed_images:

                logging.info("Processing image: %s", image_path)

                # Save the image path as completed so it isn't processed again
                completed_images.append(image_path)

                # Load the image to process
                image: consts.Image = cv2.imread(image_dir + "/" + image_path)

                # Get the camera parameters from the loaded parameter file
                camera_parameters: consts.CameraParameters = image_parameters[image_path]

                camera_parameters["altitude_f"] = (
                    camera_parameters["altitude"] * consts.FEET_PER_METER
                )
                camera_parameters["focal_length"] = 4

                new_image: consts.ImageInfo = {
                    "image_path": image_path,
                    "image": image,
                    "image_shape": {
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "channels": image.shape[2],
                    },
                    "camera_parameters": camera_parameters,
                }

                pixel_per_foot(image.shape, camera_parameters)
                map.add_img(new_image)
                cv2.imwrite("calcmap.png", map.img)

    # map.img = np.delete(map.img, 3, 2)

    map.img = map.img[:, :, :3]
    # Output final map
    cv2.imwrite("map.png", map.img[:, :, :3])
