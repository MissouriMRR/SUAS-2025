"""Implements the behavior of the ODLC state."""

import asyncio
from ctypes import c_bool
import logging
import json
from multiprocessing import Value
from multiprocessing.sharedctypes import SynchronizedBase
from pathlib import Path
import traceback

from flight.camera import Camera

from flight.extract_gps import extract_gps, GPSData
from flight.waypoint.goto import move_to
from integration_tests.emg_obj_vision import emg_integration_pipeline
from state_machine.flight_settings import FlightSettings
from state_machine.state_tracker import update_state
from state_machine.states.airdrop import Airdrop
from state_machine.states.odlc import ODLC
from state_machine.states.state import State
from vision.flyover_vision_pipeline import flyover_pipeline


async def run(self: ODLC) -> State:
    """
    Implements the run method for the ODLC state.

    This method initiates the ODLC scanning process of the drone, takes pictures and transfers
    picture data to the vision code, and then transitions to the Airdrop state.

    Parameters
    ----------
    self : ODLC
        The current instance of the ODLC state.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully scanned the ODLC area.

    Raises
    ------
    asyncio.CancelledError
        If the execution of the ODLC state is canceled.

    Notes
    -----
    The type hinting for the capture_status variable is broken, see
    https://github.com/python/typeshed/issues/8799
    """
    try:
        update_state("ODLC")
        # Syncronized type hint is broken, see https://github.com/python/typeshed/issues/8799
        capture_status: SynchronizedBase[c_bool] = Value(c_bool, False)  # type: ignore

        vision_task: asyncio.Task[None] = asyncio.ensure_future(
            vision_odlc_logic(capture_status, self.flight_settings)
        )

        flight_task: asyncio.Task[None] = asyncio.ensure_future(find_odlcs(self, capture_status))

        logging.info("Starting check for task completion")

        while not vision_task.done():
            await asyncio.sleep(0.25)

        while not flight_task.done():
            await asyncio.sleep(0.25)

        logging.info("ODLC scan complete. State completing...")
        vision_task.cancel()
        flight_task.cancel()
    except asyncio.CancelledError as ex:
        logging.error("ODLC state canceled")
        traceback.print_exc()
        raise ex

    return Airdrop(self.drone, self.flight_settings)


async def find_odlcs(self: ODLC, capture_status: "SynchronizedBase[c_bool]") -> None:
    """
    Implements the run method for the ODLC state.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully scanned the ODLC area.

    Notes
    -----
    This method is responsible for initiating the ODLC scanning process of the drone.
    """

    # Initialize the camera
    if not self.flight_settings.sim_flag:
        camera: Camera | None = Camera()
    else:
        camera = None

    # The waypoint values stored in waypoint_data.json are all that are needed
    # to traverse the whole odlc drop location
    # because it is a small rectangle
    # The first waypoint is the midpoint of
    # the left side of the rectangle(one of the short sides), the second point is the
    # midpoint of the right side of the rectangle(other short side),
    # and the third point is the top left corner of the rectangle
    # it goes there for knowing where the drone ends to travel to each of the drop locations,
    # the altitude is locked at 100 because
    # we want the drone to stay level and the camera to view the whole odlc boundary
    # the altitude 100 feet was chosen to cover the whole odlc boundary
    # because the boundary is 70ft by 360ft the fov of the camera
    # is vertical 52.1 degrees and horizontal 72.5,
    # so using the minimum length side of the photo the coverage would be 90 feet allowing
    # 10 feet overlap on both sides

    gps_data: GPSData = extract_gps(self.flight_settings.path_data_path)

    loops: int = 0  # Max amount of loops before giving up
    while loops <= 0:
        logging.info("Starting odlc zone flyover")
        loops += 1

        # traverses the 3 waypoints starting at the midpoint on left to midpoint on the right
        # then to the top left corner at the rectangle
        point: int
        for point in range(3):
            take_photos: bool = True

            if point == 0:
                logging.info("Moving to the center of the west boundary")
            elif point == 1:
                # starts taking photos at a .5 second interval because we want
                # to get multiple photos of the boundary so there is overlap and
                # the speed of the drone should be 20 m/s which is 64 feet/s which means
                # it will traverse the length of the boundary (360 ft) in 6 sec
                # and that means with the shortest length of photos
                #  being taken depending on rotation
                # would be 90 feet and we want to take multiple photos
                # so we would need a minimum of 4 photos to cover
                #  the whole boundary and we want multiple,
                # so using .5 seconds between each photo allows
                # it to take a minimum of 12 photos of
                #  the odlc boundary which will capture the whole area

                logging.info("Moving to the center of the east boundary")
                take_photos = True

            elif point == 2:
                logging.info("Moving to the north west corner")

            if camera:
                await camera.odlc_move_to(
                    self.drone,
                    gps_data["odlc_waypoints"][point].latitude,
                    gps_data["odlc_waypoints"][point].longitude,
                    gps_data["odlc_altitude"],
                    5 / 6,
                    take_photos,
                )
            else:
                await move_to(
                    self.drone.system,
                    gps_data["odlc_waypoints"][point].latitude,
                    gps_data["odlc_waypoints"][point].longitude,
                    gps_data["odlc_altitude"],
                )

        if self.flight_settings.standard_object_count <= 0:
            break

        with open("flight/data/output.json", encoding="ascii") as output:
            airdrop_dict = json.load(output)
            airdrops: int = len(airdrop_dict)

        if airdrops >= self.flight_settings.standard_object_count:
            break

    capture_status.value = c_bool(True)  # type: ignore
    self.drone.odlc_scan = False
    logging.info("ODLC scan complete")


async def vision_odlc_logic(
    capture_status: "SynchronizedBase[c_bool]", flight_settings: FlightSettings
) -> None:
    """
    Implements the run method for the ODLC state.

    Parameters
    ----------
    capture_status : SynchronizedBase[c_bool]
        A text file containing True if all images have been taken and False otherwise
    flight_settings : FlightSettings
        Settings for this flight.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully scanned the ODLC area.

    Notes
    -----
    This method is responsible for initiating the ODLC scanning process of the drone
    and transitioning it to the Airdrop state.
    """
    camera_data_filename: str = "flight/data/camera.json"

    pipeline = (
        emg_integration_pipeline if flight_settings.standard_object_count == 0 else flyover_pipeline
    )

    # Wait until camera.json exists
    logging.info("Waiting for %s to exist", camera_data_filename)
    while not Path(camera_data_filename).is_file():
        await asyncio.sleep(1)
    logging.info("Camera data file found.")

    await pipeline("flight/data/camera.json", capture_status, "flight/data/output.json")


# Setting the run_callable attribute of the ODLC class to the run function
ODLC.run_callable = run
