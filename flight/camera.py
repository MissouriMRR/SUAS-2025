"""A class that contains all the needed camera functionality for the drone."""

# pylint: disable=too-many-locals
# Not worth to deal with this with the time crunch we are in

import asyncio
import json
import logging
import os
from datetime import datetime

import gphoto2

from flight.waypoint.calculate_distance import calculate_distance
from state_machine.drone import Drone

WAYPOINT_TOLERANCE: int = 1  # in meters


class Camera:
    """
    Initialize a new Camera object to control the Sony RX100-VII camera on the drone

    Attributes
    ----------
    camera : gphoto2.Camera
        The gphoto2 camera object.
    session_id : int
        The session id for the current session.
        This will start at 0 the first time pictures are taken on a given day.
        Will then increment by 1 for each session on a given day.
    image_id : int
        The image id for the current image.
        Starts at 0 and increments by 1 for each image taken.

    Methods
    -------
    capture_photo(path: str = f"{os.getcwd()}/images/")
        Capture a photo and save it to the specified path.
        The default path is the images folder in the current working directory.
        The file name will be the file format attribute.
        Returns the file name and the file path.
    odlc_move_to(
        drone: Drone,
        latitude: float,
        longitude: float,
        altitude: float,
        fast_param: float,
        take_photos: float
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes photos along the way if take_photos is True.
    """

    def __init__(self) -> None:
        self.camera: gphoto2.Camera = gphoto2.Camera()
        self.camera.init()

        self.session_id: int = 0
        if os.path.exists(f"{os.getcwd()}/images/"):
            for file in os.listdir(f"{os.getcwd()}/images/"):
                if file.startswith(f"{datetime.now().strftime('%Y%m%d')}"):
                    if int(file.split("_")[1]) >= self.session_id:
                        self.session_id = int(file.split("_")[1]) + 1

        self.image_id: int = 0

        logging.info("Camera initialized")

    async def capture_photo(self, path: str = f"{os.getcwd()}/images/") -> tuple[str, str]:
        """
        Capture a photo and save it to the specified path.

        Parameters
        ----------
        path : str, optional
            The path to save the image to, by default f"{os.getcwd()}/images/"


        Returns
        -------
        tuple[str, str]
            The file name and the file path.
        """
        # If the images folder doesn't exist, we can't save images.
        # So we have to make sure the images folder exists.
        os.makedirs(path, mode=0o777, exist_ok=True)

        file_path = self.camera.capture(gphoto2.GP_CAPTURE_IMAGE)
        while True:
            event_type, _event_data = self.camera.wait_for_event(100)
            if event_type == gphoto2.GP_EVENT_CAPTURE_COMPLETE:
                photo_name: str = (
                    f"{datetime.now().strftime('%Y%m%d')}_{self.session_id}_{self.image_id:04d}.jpg"
                )

                cam_file = gphoto2.check_result(
                    gphoto2.gp_camera_file_get(
                        self.camera,
                        file_path.folder,
                        file_path.name,
                        gphoto2.GP_FILE_TYPE_NORMAL,
                    )
                )
                target_name: str = f"{path}{photo_name}"
                cam_file.save(target_name)
                self.image_id += 1
                logging.info("Image is being saved to %s", target_name)
                return target_name, photo_name

    async def odlc_move_to(
        self,
        drone: Drone,
        latitude: float,
        longitude: float,
        altitude: float,
        take_photos: bool,
        heading: float = 0,
    ) -> None:
        """
        This function takes in a latitude, longitude and altitude and autonomously
        moves the drone to that waypoint. This function will also auto convert the altitude
        from feet to meters. It will take photos along the path if passed true in take_photos and
        add the point and name of photo to a json

        Parameters
        ----------
        drone: System
            a drone object that has all offboard data needed for computation
        latitude: float
            a float containing the requested latitude to move to
        longitude: float
            a float containing the requested longitude to move to
        altitude: float
            a float contatining the requested altitude to go to in meters
        fast_param: float
            a float that determines if the drone will take less time checking its precise location
            before moving on to another waypoint. If its 1, it will move at normal speed,
            if its less than 1(0.83), it will be faster.
        take_photos: bool
            will take photos with the camera until the position has been reached
        """
        if take_photos:
            await drone.system.action.set_maximum_speed(5)

        info: dict[str, dict[str, int | list[int | float] | float]] = {}

        # get current altitude
        async for terrain_info in drone.system.telemetry.home():
            absolute_altitude: float = terrain_info.absolute_altitude_m
            break

        await drone.system.action.goto_location(
            latitude, longitude, altitude + absolute_altitude, heading
        )
        location_reached: bool = False
        # First determine if we need to move fast through waypoints or need to slow down at each one
        # Then loops until the waypoint is reached
        while not location_reached:
            logging.info("Going to waypoint")
            async for position in drone.system.telemetry.position():
                # continuously checks current latitude, longitude and altitude of the drone
                drone_lat: float = position.latitude_deg
                drone_long: float = position.longitude_deg
                drone_alt: float = position.relative_altitude_m

                total_distance: float = calculate_distance(
                    drone_lat, drone_long, drone_alt, latitude, longitude, altitude
                )

                if total_distance < WAYPOINT_TOLERANCE:  # within 1 meter of the point
                    location_reached = True
                    logging.info("Arrived %sm away from waypoint", total_distance)
                    break

            await asyncio.sleep(2)

            if take_photos:
                _full_path: str
                file_path: str
                _full_path, file_path = await self.capture_photo()

                async for euler in drone.system.telemetry.attitude_euler():
                    roll_deg: float = euler.roll_deg
                    pitch_deg: float = euler.pitch_deg
                    yaw_deg: float = euler.yaw_deg
                    break

                point: dict[str, dict[str, int | list[int | float] | float]] = {
                    file_path: {
                        "focal_length": 24,
                        "rotation_deg": [
                            roll_deg,
                            pitch_deg,
                            yaw_deg,
                        ],
                        "drone_coordinates": [latitude, longitude],
                        "altitude_f": drone_alt,
                    }
                }

                info.update(point)

                current_photos: dict[str, dict[str, int | list[int | float] | float]] = {}
                if os.path.exists("flight/data/camera.json"):
                    with open("flight/data/camera.json", "r", encoding="utf8") as current_data:
                        try:
                            current_photos = json.load(current_data)
                        except json.JSONDecodeError:
                            pass

                with open("flight/data/camera.json", "w", encoding="ascii") as camera:
                    json.dump(current_photos | info, camera)

                await drone.system.action.set_maximum_speed(13.41)
            # tell machine to sleep to prevent constant polling, preventing battery drain
            await asyncio.sleep(1)
        return
