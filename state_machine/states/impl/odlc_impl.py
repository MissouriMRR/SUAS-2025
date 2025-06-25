"""Implements the behavior of the ODLC state."""

import asyncio
import logging
from pathlib import Path
import traceback

from flight.camera import CameraIRL, CameraAirSim
from flight.extract_gps import extract_gps, GPSData, OdlcWaypoint
from flight.waypoint.goto import move_to
from state_machine.flight_settings import SimMode
from state_machine.state_tracker import (
    update_state,
    update_drone,
    update_flight_settings,
)
from state_machine.states.mapping import Mapping
from state_machine.states.odlc import ODLC
from state_machine.states.state import State

from vision.vision_pipeline import flyover_pipeline
from vision.common import camera_config


async def run(self: ODLC) -> State:
    """
    Implements the run method for the ODLC state.

    This method initiates the ODLC scanning process of the drone, takes pictures and transfers
    picture data to the vision code, and then transitions to the Mapping state.

    Parameters
    ----------
    self : ODLC
        The current instance of the ODLC state.

    Returns
    -------
    Mapping : State
        The next state after the drone has successfully scanned the ODLC area.

    Raises
    ------
    asyncio.CancelledError
        If the execution of the ODLC state is canceled.
    """

    camera_config.update_sim_mode(self.flight_settings.sim_mode)

    if self.flight_settings.skip_odlc_and_airdrop:
        return Mapping(self.drone, self.flight_settings)

    try:
        update_state("ODLC")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)
        logging.info("ODLC state running")

        capture_status: asyncio.Event = asyncio.Event()

        asyncio.ensure_future(vision_odlc_logic(self, capture_status))

        flight_task: asyncio.Task[None] = asyncio.ensure_future(find_odlcs(self, capture_status))

        logging.info("Starting check for flight task completion")

        while not flight_task.done():
            await asyncio.sleep(0.25)

        logging.info("ODLC flight scan complete. State completing...")
        flight_task.cancel()
    except asyncio.CancelledError as ex:
        logging.error("ODLC state canceled")
        traceback.print_exc()
        raise ex

    return Mapping(self.drone, self.flight_settings)


async def find_odlcs(self: ODLC, capture_status: asyncio.Event) -> None:
    """
    Implements the flight logic fro the run method of the ODLC state.

    Parameters
    ----------
    self : ODLC
        The ODLC state object.
    capture_status : asyncio.Event
        An event that is set when the drone has successfully captured all images.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully scanned the ODLC area.

    Notes
    -----
    This method is responsible for initiating the ODLC scanning process of the drone.
    """

    # Initialize the camera
    if self.flight_settings.sim_mode is SimMode.REAL:
        camera: CameraIRL | CameraAirSim | None = CameraIRL()
        # The gimbal takes some time to adjust its angle, wait for it to be ready
        await asyncio.sleep(3)
    elif self.flight_settings.sim_mode is SimMode.AIRSIM:
        camera = CameraAirSim()
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

    gps_data: GPSData = extract_gps(self.flight_settings.mission_data_path)

    logging.info("Starting odlc zone flyover")

    # Go to first point quickly and without taking pictures
    first_point = gps_data["odlc_waypoints"][0]
    await move_to(
        self.drone.vehicle,
        first_point.latitude,
        first_point.longitude,
        gps_data["odlc_altitude"],
        airspeed=25,
    )

    # traverses the 3 waypoints starting at the midpoint on left to midpoint on the right
    # then to the top left corner at the rectangle
    i: int
    point: OdlcWaypoint
    for i, point in enumerate(gps_data["odlc_waypoints"], start=1):
        take_photos: bool = True

        logging.info("Moving to ODLC scan point %d", i)

        if camera is not None:
            await camera.odlc_move_to(
                self.drone.vehicle,
                point.latitude,
                point.longitude,
                gps_data["odlc_altitude"],
                take_photos,
            )
        else:
            await move_to(
                self.drone.vehicle,
                point.latitude,
                point.longitude,
                gps_data["odlc_altitude"],
                airspeed=5,
            )

    if camera:
        camera.disconnect()
    capture_status.set()
    self.drone.odlc_scan = False
    logging.info("ODLC scan complete")


async def vision_odlc_logic(self: ODLC, capture_status: asyncio.Event) -> None:
    """
    Implements the vision logic for the run method of the ODLC state.

    Parameters
    ----------
    self : ODLC
        The ODLC state object.
    capture_status : asyncio.Event
        An event that is set when the drone has successfully captured all images.
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

    # Wait until camera.json exists
    logging.info("Waiting for %s to exist", camera_data_filename)
    while not Path(camera_data_filename).is_file():
        await asyncio.sleep(1)
    logging.info("Camera data file found.")

    await flyover_pipeline(
        self.flight_settings,
        "flight/data/camera.json",
        capture_status,
        "flight/data/output.json",
    )


# Setting the run_callable attribute of the ODLC class to the run function
ODLC.run_callable = run
