"""Checks if the camera object is connected correctly and can save pictures to the images folder."""

import asyncio
from datetime import datetime
import logging
import os

import gphoto2


async def test_capture_image(photo_num: int = 1) -> None:
    """Test the capture_photo method of the Camera class.

    Parameters
    ----------
    photo_num : int, optional
        The number of photos to take, by default 1
    """
    logging.info("Connecting to camera...")
    camera: gphoto2.Camera = gphoto2.Camera()
    try:
        camera.init()
    except gphoto2.GPhoto2Error as ex:
        logging.error("Failed to initialize camera. Exiting...")
        logging.error(ex)
        return

    logging.info("Camera connected. Initializing capture...")
    session_id: int = 0
    if os.path.exists(f"{os.getcwd()}/images/"):
        for file in os.listdir(f"{os.getcwd()}/images/"):
            if file.startswith(f"{datetime.now().strftime('%Y%m%d')}"):
                if int(file.split("_")[1]) >= session_id:
                    session_id = int(file.split("_")[1]) + 1

    image_id: int = 1
    logging.info("Capturing images...")

    # If the images folder doesn't exist, we can't save images.
    # So we have to make sure the images folder exists.
    path: str = f"{os.getcwd()}/images/"
    os.makedirs(path, mode=0o777, exist_ok=True)

    file_path = camera.capture(gphoto2.GP_CAPTURE_IMAGE)
    while image_id <= photo_num:
        while True:
            event_type, _event_data = camera.wait_for_event(100)
            if event_type == gphoto2.GP_EVENT_CAPTURE_COMPLETE:
                photo_name: str = (
                    f"{datetime.now().strftime('%Y%m%d')}_{session_id}_{image_id:04d}.jpg"
                )

                cam_file = gphoto2.check_result(
                    gphoto2.gp_camera_file_get(
                        camera,
                        file_path.folder,
                        file_path.name,
                        gphoto2.GP_FILE_TYPE_NORMAL,
                    )
                )
                target_name: str = f"{path}{photo_name}"
                cam_file.save(target_name)
                image_id += 1
                logging.info("Image #%d is being saved to %s", image_id, target_name)
                continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_capture_image())
