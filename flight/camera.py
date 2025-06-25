"""A class that contains all the needed camera functionality for the drone."""

# pylint: disable=too-many-locals
# Not worth to deal with this with the time crunch we are in
import asyncio
from datetime import datetime
import json
import logging
import math
import os
from typing import Any, TextIO
from abc import ABC, abstractmethod


import aiofiles  # type: ignore
import aiohttp
import dronekit
from siyi_sdk.siyi_sdk import SIYISDK
import airsim

from flight.waypoint.goto import move_to
from flight.waypoint.calculate_distance import calculate_distance
from vision.common.constants import CameraParameters


WAYPOINT_TOLERANCE: int = 1  # in meters


class Camera(ABC):
    """
    Initialize a new Camera object to control the SIYI A8 Mini gimbal camera on the drone

    Attributes
    ----------
    camera : SIYISDK
        The object that controls the camera.
    session_id : int
        The session id for the current session.
        This will start at 0 the first time pictures are taken on a given day.
        Will then increment by 1 for each session on a given day.
    image_id : int
        The image id for the current image.
        Starts at 0 and increments by 1 for each image taken.
    base_api_url : str
        The base API URL for the camera

    Methods
    -------
    capture_photo(path: str = f"{os.getcwd()}/images/")
        Capture a photo and save it to the specified path.
        The default path is the images folder in the current working directory.
        The file name will be the file format attribute.
        Returns the file name and the file path.
    mapping_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes photos along the way.
    odlc_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        take_photos: bool
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes a photo at the end if take_photos is True.
    _get_camera_parameters(drone: dronekit.Vehicle)
        Get the current camera information to associate with a photo.
    """

    def __init__(self) -> None:
        self.image_id: int = 0
        self.base_api_url: str | None = None
        self.camera: SIYISDK | None = None
        self.session_id: int = 0

    @abstractmethod
    async def capture_photo(self, path: str = f"{os.getcwd()}/images/") -> tuple[str, str] | None:
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
        return None

    @abstractmethod
    async def mapping_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float,
    ) -> None:
        """
        Moves to the drone to the requested waypoint while taking photos for the mapping state.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone object with the camera.
        latitude : float
            The requested latitude to move to, in degrees.
        longitude : float
            The requested longitude to move to, in degrees.
        altitude : float
            The requested altitude to go to, in meters.
        interval : float
            The interval, in meters, at which to take photos.
        """
        return None

    @abstractmethod
    async def odlc_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        take_photos: bool,
    ) -> None:
        """
        This function takes in a latitude, longitude and altitude and autonomously
        moves the drone to that waypoint. It will take a photo at the end if passed
        True in take_photos and add the point and name of photo to a json.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone object with the camera.
        latitude : float
            The requested latitude to move to, in degrees.
        longitude : float
            The requested longitude to move to, in degrees.
        altitude : float
            The requested altitude to go to, in meters.
        take_photos : bool
            Whether to take a photo once the waypoint has been reached.
        """
        return None

    @abstractmethod
    async def _get_camera_parameters(self, drone: dronekit.Vehicle) -> CameraParameters | None:
        """
        Gets the current camera information based on a drone's position_pitch.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone whose position will be stored in the result.

        Returns
        -------
        CameraParameters
            Camera information to be associated with a photo.
        """
        return None

    def disconnect(self) -> None:
        """
        Disconnects the IRL Camera.
        """
        return None


class CameraIRL(Camera):
    """
    Initialize a new Camera object to control the SIYI A8 Mini gimbal camera on the drone

    Attributes
    ----------
    camera : SIYISDK
        The object that controls the camera.
    session_id : int
        The session id for the current session.
        This will start at 0 the first time pictures are taken on a given day.
        Will then increment by 1 for each session on a given day.
    image_id : int
        The image id for the current image.
        Starts at 0 and increments by 1 for each image taken.
    base_api_url : str
        The base API URL for the camera

    Methods
    -------
    capture_photo(path: str = f"{os.getcwd()}/images/")
        Capture a photo and save it to the specified path.
        The default path is the images folder in the current working directory.
        The file name will be the file format attribute.
        Returns the file name and the file path.
    mapping_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes photos along the way.
    odlc_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        take_photos: bool
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes a photo at the end if take_photos is True.
    _get_camera_parameters(drone: dronekit.Vehicle)
        Get the current camera information to associate with a photo.
    """

    def __init__(self) -> None:
        super().__init__()
        logging.info("Connecting to camera...")
        self.camera: SIYISDK = SIYISDK()
        if not self.camera.connect(maxRetries=10):
            logging.error("Failed to connect to the camera")

        self.session_id: int = 0
        if os.path.exists(f"{os.getcwd()}/images/"):
            for file in os.listdir(f"{os.getcwd()}/images/"):
                if file.startswith(f"{datetime.now().strftime('%Y%m%d')}"):
                    if int(file.split("_")[1]) >= self.session_id:
                        self.session_id = int(file.split("_")[1]) + 1

        self.image_id: int = 0
        self.base_api_url: str = "http://192.168.144.25:82//cgi-bin/media.cgi"

        # Set gimbal to point straight down for taking pictures
        if not self.camera.requestSetAngles(0, -90):
            logging.error("Failed to set gimbal angle")

        logging.info("IRL Camera initialized")

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

        took_photo: bool = self.camera.requestPhoto()
        if not took_photo:
            logging.error("Failed to take photo")
            raise ValueError("Failed to take photo")

        # Need to wait a bit after sending the capture command to allow the
        # gimbal to process/save the photo, else we run into a race condition
        await asyncio.sleep(0.5)

        # Retrieve the image from the gimbal SD card
        session: aiohttp.ClientSession
        async with aiohttp.ClientSession() as session:
            response: aiohttp.client._RequestContextManager
            json_data: dict[str, Any]

            async with session.get(
                f"{self.base_api_url}/api/v1/getmediacount?media_type=0&path=101SIYI_IMG"
            ) as response:
                json_data = await response.json()
                if response.status != 200:
                    logging.error("Failed to get media count. Response data: %s", json_data)
                    raise ValueError("Failed to get media count")
                media_count: int = json_data["data"]["count"]

            # Have to request for all images in the directory due to a SIYI firmware bug
            async with session.get(
                f"{self.base_api_url}/api/v1/getmedialist?"
                f"media_type=0&path=101SIYI_IMG&start=0&count={media_count}"
            ) as response:
                json_data = await response.json()
                if response.status != 200:
                    logging.error("Failed to get media list. Response data: %s", json_data)
                    raise ValueError("Failed to get media list")
                try:
                    media_path: str = json_data["data"]["list"][-1]["url"]
                except IndexError as exc:
                    logging.error("Failed to get media path. Response data: %s", json_data)
                    raise ValueError("Failed to get media path") from exc

            async with session.get(media_path) as response:
                binary_data: bytes = await response.read()
                if response.status != 200:
                    logging.error("Failed to get media. Response data: %s", binary_data)
                    raise ValueError("Failed to get media")
                photo_name: str = (
                    f"{datetime.now().strftime('%Y%m%d')}_{self.session_id}_{self.image_id:04d}.jpg"
                )
                target_name: str = f"{path}{photo_name}"

                file: aiofiles.base.AiofilesContextManager
                async with aiofiles.open(target_name, "wb") as file:
                    await file.write(binary_data)

                logging.info("Image #%d is being saved to %s", self.image_id, target_name)
                self.image_id += 1
                return target_name, photo_name

    async def mapping_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float,
    ) -> None:
        """
        Moves to the drone to the requested waypoint while taking photos for the mapping state.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone object with the camera.
        latitude : float
            The requested latitude to move to, in degrees.
        longitude : float
            The requested longitude to move to, in degrees.
        altitude : float
            The requested altitude to go to, in meters.
        interval : float
            The interval, in meters, at which to take photos.
        heading : float, default 0
            The yaw in which the camera should point, in degrees (0 is north, 90 is west).
        """
        info: dict[str, CameraParameters] = {}

        self.camera.requestSetAngles(0, -90)  # Point straight down if it isn't already

        camera_parameters: CameraParameters = await self._get_camera_parameters(drone)
        file_path: str
        _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
        point: dict[str, CameraParameters] = {file_path: camera_parameters}
        info.update(point)

        goto_task: asyncio.Task[None] = asyncio.ensure_future(
            move_to(
                drone,
                latitude,
                longitude,
                altitude,
                airspeed=10,
                tolerance=WAYPOINT_TOLERANCE,
            )
        )

        start_pos: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

        start_lat: float = start_pos.lat
        start_lon: float = start_pos.lon
        start_alt: float = start_pos.alt

        next_interval_count: int = 1
        while not goto_task.done():
            position: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

            drone_lat: float = position.lat
            drone_long: float = position.lon
            drone_alt: float = position.alt

            distance: float = calculate_distance(
                drone_lat, drone_long, drone_alt, start_lat, start_lon, start_alt
            )

            if distance >= next_interval_count * interval:
                next_interval_count += 1
                camera_parameters = await self._get_camera_parameters(drone)
                _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
                point = {file_path: camera_parameters}
                info.update(point)

            await asyncio.sleep(0.25)
        await asyncio.sleep(1.0)

        camera_parameters = await self._get_camera_parameters(drone)
        _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
        point = {file_path: camera_parameters}
        info.update(point)

        current_photos: dict[str, CameraParameters] = {}
        if os.path.exists("flight/data/mapping_photos.json"):
            current_data: TextIO
            with open("flight/data/mapping_photos.json", "r", encoding="utf8") as current_data:
                try:
                    current_photos = json.load(current_data)
                except json.JSONDecodeError:
                    pass

        with open("flight/data/mapping_photos.json", "w", encoding="ascii") as camera:
            json.dump(current_photos | info, camera)

    async def odlc_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        take_photos: bool,
    ) -> None:
        """
        This function takes in a latitude, longitude and altitude and autonomously
        moves the drone to that waypoint. It will take a photo at the end if passed
        True in take_photos and add the point and name of photo to a json.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone object with the camera.
        latitude : float
            The requested latitude to move to, in degrees.
        longitude : float
            The requested longitude to move to, in degrees.
        altitude : float
            The requested altitude to go to, in meters.
        take_photos : bool
            Whether to take a photo once the waypoint has been reached.
        heading : float, default 0
            The yaw in which the camera should point, in degrees (0 is north, 90 is west).
        """
        info: dict[str, CameraParameters] = {}

        goto_task: asyncio.Task[None] = asyncio.ensure_future(
            move_to(
                drone,
                latitude,
                longitude,
                altitude,
                airspeed=5.0,
                tolerance=WAYPOINT_TOLERANCE,
            )
        )

        if take_photos:
            # Point the gimbal straight down
            self.camera.requestSetAngles(0, -90)

        while not goto_task.done():
            if not take_photos:
                await asyncio.sleep(1)
                continue

            camera_parameters: CameraParameters = await self._get_camera_parameters(drone)
            file_path: str
            _, file_path = await self.capture_photo()
            point: dict[str, CameraParameters] = {file_path: camera_parameters}
            info.update(point)

        current_photos: dict[str, CameraParameters] = {}
        if os.path.exists("flight/data/camera.json"):
            current_data: TextIO
            with open("flight/data/camera.json", "r", encoding="utf8") as current_data:
                try:
                    current_photos = json.load(current_data)
                except json.JSONDecodeError:
                    pass

        camera: TextIO
        with open("flight/data/camera.json", "w", encoding="ascii") as camera:
            json.dump(current_photos | info, camera)

    async def _get_camera_parameters(self, drone: dronekit.Vehicle) -> CameraParameters:
        """
        Gets the current camera information based on a drone's position_pitch.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone whose position will be stored in the result.

        Returns
        -------
        CameraParameters
            Camera information to be associated with a photo.
        """
        location: dronekit.LocationGlobalRelative = drone.location.global_relative_frame
        gimbal_attitude = self.camera.getAttitude()

        attitude: dronekit.Attitude = drone.attitude
        roll_deg: float = math.degrees(attitude.roll) + gimbal_attitude[2]
        pitch_deg: float = math.degrees(attitude.pitch) + gimbal_attitude[1]
        yaw_deg: float = math.degrees(attitude.yaw) + gimbal_attitude[0]

        return CameraParameters(
            rotation_deg=[roll_deg, pitch_deg, yaw_deg],
            drone_coordinates=[location.lat, location.lon],
            altitude=location.alt,
        )

    def disconnect(self) -> None:
        """
        Disconnects the IRL Camera.
        """
        self.camera.disconnect()


class CameraAirSim(Camera):
    """
    Initialize a new Camera object to control a simulated airsim camera

    Attributes
    ----------
    client : airsim.MultirotorClient
        The simulated drone.
    session_id : int
        The session id for the current session.
        This will start at 0 the first time pictures are taken on a given day.
        Will then increment by 1 for each session on a given day.
    image_id : int
        The image id for the current image.
        Starts at 0 and increments by 1 for each image taken.
    base_api_url : str
        The base API URL for the camera


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
        take_photos: float
    )
    mapping_move_to(
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float
    )
        Move the drone to the specified latitude, longitude, and altitude.
        Takes photos along the way if take_photos is True.
    _get_camera_parameters(drone: dronekit.Vehicle)
        Get the current camera information to associate with a photo.
    """

    def __init__(self) -> None:
        super().__init__()

        self.client: airsim.MultirotorClient = airsim.MultirotorClient()

        self.session_id: int = 0
        if os.path.exists(f"{os.getcwd()}/images/"):
            for file in os.listdir(f"{os.getcwd()}/images/"):
                if file.startswith(f"{datetime.now().strftime('%Y%m%d')}"):
                    if int(file.split("_")[1]) >= self.session_id:
                        self.session_id = int(file.split("_")[1]) + 1

        self.image_id: int = 0
        self.base_api_url: str = "http://"

        logging.info("Airsim Camera initialized")

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
        os.makedirs(path, mode=0o777, exist_ok=True)
        cam_file = self.client.simGetImage("bottom_center", airsim.ImageType.Scene)
        photo_name: str = (
            f"{datetime.now().strftime('%Y%m%d')}_{self.session_id}_{self.image_id:04d}.jpg"
        )
        target_name: str = f"{path}{photo_name}"
        async with aiofiles.open(target_name, mode="wb") as file:
            await file.write(cam_file)

        self.image_id += 1
        logging.info("Image #%d is being saved to %s", self.image_id, target_name)
        return target_name, photo_name

    async def mapping_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        interval: float,
    ) -> None:
        """
        Moves to the drone to the requested waypoint while taking photos for the mapping state.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone object with the camera.
        latitude : float
            The requested latitude to move to, in degrees.
        longitude : float
            The requested longitude to move to, in degrees.
        altitude : float
            The requested altitude to go to, in meters.
        interval : float
            The interval, in meters, at which to take photos.
        heading : float, default 0
            The yaw in which the camera should point, in degrees (0 is north, 90 is west).
        """
        info: dict[str, CameraParameters] = {}

        camera_parameters: CameraParameters = await self._get_camera_parameters(drone)
        file_path: str
        _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
        point: dict[str, CameraParameters] = {file_path: camera_parameters}
        info.update(point)

        goto_task: asyncio.Task[None] = asyncio.ensure_future(
            move_to(
                drone,
                latitude,
                longitude,
                altitude,
                airspeed=5.0,
                tolerance=WAYPOINT_TOLERANCE,
            )
        )

        start_pos: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

        start_lat: float = start_pos.lat
        start_lon: float = start_pos.lon
        start_alt: float = start_pos.alt

        next_interval_count: int = 1
        while not goto_task.done():
            position: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

            drone_lat: float = position.lat
            drone_long: float = position.lon
            drone_alt: float = position.alt

            distance: float = calculate_distance(
                drone_lat, drone_long, drone_alt, start_lat, start_lon, start_alt
            )

            if distance >= next_interval_count * interval:
                next_interval_count += 1
                camera_parameters = await self._get_camera_parameters(drone)
                _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
                point = {file_path: camera_parameters}
                info.update(point)

            await asyncio.sleep(0.25)
        await asyncio.sleep(1.0)

        camera_parameters = await self._get_camera_parameters(drone)
        _, file_path = await self.capture_photo(f"{os.getcwd()}/mapping_images/")
        point = {file_path: camera_parameters}
        info.update(point)

        current_photos: dict[str, CameraParameters] = {}
        if os.path.exists("flight/data/mapping_photos.json"):
            current_data: TextIO
            with open("flight/data/mapping_photos.json", "r", encoding="utf8") as current_data:
                try:
                    current_photos = json.load(current_data)
                except json.JSONDecodeError:
                    pass

        with open("flight/data/mapping_photos.json", "w", encoding="ascii") as camera:
            json.dump(current_photos | info, camera)

    async def odlc_move_to(
        self,
        drone: dronekit.Vehicle,
        latitude: float,
        longitude: float,
        altitude: float,
        take_photos: bool,
    ) -> None:
        """
        This function takes in a latitude, longitude and altitude and autonomously
        moves the drone to that waypoint. It will take a photo at the end if passed
        True in take_photos and add the point and name of photo to a json.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone object with the camera.
        latitude : float
            The requested latitude to move to, in degrees.
        longitude : float
            The requested longitude to move to, in degrees.
        altitude : float
            The requested altitude to go to, in meters.
        take_photos : bool
            Whether to take a photo once the waypoint has been reached.
        heading : float, default 0
            The yaw in which the camera should point, in degrees (0 is north, 90 is west).
        """
        info: dict[str, CameraParameters] = {}

        await move_to(
            drone,
            latitude,
            longitude,
            altitude,
            airspeed=5.0,
            tolerance=WAYPOINT_TOLERANCE,
        )

        await asyncio.sleep(2)

        if not take_photos:
            return

        camera_parameters: CameraParameters = await self._get_camera_parameters(drone)

        _, file_path = await self.capture_photo()
        point: dict[str, CameraParameters] = {file_path: camera_parameters}
        info.update(point)

        current_photos: dict[str, CameraParameters] = {}
        if os.path.exists("flight/data/camera.json"):
            current_data: TextIO
            with open("flight/data/camera.json", "r", encoding="utf8") as current_data:
                try:
                    current_photos = json.load(current_data)
                except json.JSONDecodeError:
                    pass

        camera: TextIO
        with open("flight/data/camera.json", "w", encoding="ascii") as camera:
            json.dump(current_photos | info, camera)

            # tell machine to sleep to prevent constant polling, preventing battery drain
            await asyncio.sleep(1)
        return

    async def _get_camera_parameters(self, drone: dronekit.Vehicle) -> CameraParameters:
        """
        Gets the current camera information based on a drone's position_pitch.

        Parameters
        ----------
        drone : dronekit.Vehicle
            The drone whose position will be stored in the result.

        Returns
        -------
        CameraParameters
            Camera information to be associated with a photo.
        """
        location: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

        attitude: dronekit.Attitude = drone.attitude
        roll_deg: float = math.degrees(attitude.roll)
        pitch_deg: float = math.degrees(attitude.pitch) - 90
        yaw_deg: float = math.degrees(attitude.yaw)

        return CameraParameters(
            rotation_deg=[roll_deg, pitch_deg, yaw_deg],
            drone_coordinates=[location.lat, location.lon],
            altitude=location.alt,
        )
