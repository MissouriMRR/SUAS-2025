"""Implements the behavior of the Mapping state."""

# pylint: disable=too-many-locals

import asyncio
import logging
import math
from typing import Final

import utm

from flight.camera import CameraIRL, CameraAirSim
from flight.extract_gps import extract_gps, GPSData
from flight.extract_gps import BoundaryPointUtm
from flight.waypoint.goto import move_to
from state_machine.flight_settings import SimMode

from state_machine.state_tracker import (
    update_state,
    update_drone,
    update_flight_settings,
)
from state_machine.states.airdrop import Airdrop
from state_machine.states.mapping import Mapping
from state_machine.states.state import State
from vision.common import camera_config

# These should be moved to a constants file
MAPPING_ALTITUDE: Final[float] = 30  # meters
HORIZONTAL_PHOTO_SPACING: Final[float] = 15  # meters
VERTICAL_PHOTO_SPACING: Final[float] = 15  # meters
MAPPING_AIRSPEED: Final[float] = 10  # meters per second


async def run(self: Mapping) -> State:
    """
    Implements the run method for the Mapping state.

    This method captures photos of the mapping area and then transitions to the Airdrop state.
    If YOLO inference from the ODLC state is still running, the code will wait for it to finish
    before transitioning to the Airdrop state.

    Returns
    -------
    Airdrop : State
        The next state after the drone has successfully landed.
    """

    camera_config.update_sim_mode(self.flight_settings.sim_mode)

    try:
        update_state("Mapping")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)

        logging.info("Mapping")

        gps_dict: GPSData = extract_gps(self.flight_settings.mission_data_path)
        mapping_boundary_utm: list[BoundaryPointUtm] = gps_dict["mapping_boundary_utm"]

        # The mapping area should be roughly rectangular with 4 vertices
        # We are going to travel in line segments parallel to the long edges

        is_long_edge_first: bool = math.hypot(
            mapping_boundary_utm[1].easting - mapping_boundary_utm[0].easting,
            mapping_boundary_utm[1].northing - mapping_boundary_utm[0].northing,
        ) >= math.hypot(
            mapping_boundary_utm[2].easting - mapping_boundary_utm[0].easting,
            mapping_boundary_utm[2].northing - mapping_boundary_utm[0].northing,
        )

        # Ensure the first edge is long
        if not is_long_edge_first:
            mapping_boundary_utm[1], mapping_boundary_utm[3] = (
                mapping_boundary_utm[3],
                mapping_boundary_utm[1],
            )

        # Take the max because it's better to take too many photos than too few
        short_edge_length = max(
            math.hypot(
                mapping_boundary_utm[3].easting - mapping_boundary_utm[0].easting,
                mapping_boundary_utm[3].northing - mapping_boundary_utm[0].northing,
            ),
            math.hypot(
                mapping_boundary_utm[2].easting - mapping_boundary_utm[1].easting,
                mapping_boundary_utm[2].northing - mapping_boundary_utm[1].northing,
            ),
        )

        # Get average direction of short edges
        step_count: int = math.ceil(short_edge_length / VERTICAL_PHOTO_SPACING)

        utm_zone_number = mapping_boundary_utm[0].zone_number
        utm_zone_letter = mapping_boundary_utm[0].zone_letter
        if self.flight_settings.sim_mode is SimMode.REAL:
            camera: CameraIRL | CameraAirSim | None = CameraIRL()
        elif self.flight_settings.sim_mode is SimMode.AIRSIM:
            camera = CameraAirSim()
        else:
            camera = None

        reverse_direction: bool = False

        # Go to first point without taking pictures
        first_easting: float
        first_northing: float
        if not reverse_direction:
            first_easting = mapping_boundary_utm[0].easting
            first_northing = mapping_boundary_utm[0].northing
        else:
            first_easting = mapping_boundary_utm[3].easting
            first_northing = mapping_boundary_utm[3].northing

        first_lat: float
        first_lon: float
        first_lat, first_lon = utm.to_latlon(
            first_easting, first_northing, utm_zone_number, utm_zone_letter
        )
        await move_to(self.drone.vehicle, first_lat, first_lon, MAPPING_ALTITUDE, airspeed=25)

        for i in range(step_count + 1):
            lerp_t = i / step_count

            # Linearly interpolate along the short edges
            start_easting: float = (1 - lerp_t) * mapping_boundary_utm[
                0
            ].easting + lerp_t * mapping_boundary_utm[3].easting
            start_northing: float = (1 - lerp_t) * mapping_boundary_utm[
                0
            ].northing + lerp_t * mapping_boundary_utm[3].northing
            end_easting: float = (1 - lerp_t) * mapping_boundary_utm[
                1
            ].easting + lerp_t * mapping_boundary_utm[2].easting
            end_northing: float = (1 - lerp_t) * mapping_boundary_utm[
                1
            ].northing + lerp_t * mapping_boundary_utm[2].northing

            # Move in a serpentine pattern
            if reverse_direction:
                start_easting, end_easting = end_easting, start_easting
                start_northing, end_northing = end_northing, start_northing

            lat: float
            lon: float
            lat, lon = utm.to_latlon(
                start_easting, start_northing, utm_zone_number, utm_zone_letter
            )
            await move_to(self.drone.vehicle, lat, lon, MAPPING_ALTITUDE, airspeed=MAPPING_AIRSPEED)

            lat, lon = utm.to_latlon(end_easting, end_northing, utm_zone_number, utm_zone_letter)

            if camera is not None:
                await camera.mapping_move_to(
                    self.drone.vehicle, lat, lon, MAPPING_ALTITUDE, HORIZONTAL_PHOTO_SPACING
                )

            reverse_direction = not reverse_direction
        if camera is not None and camera.camera is not None:
            camera.camera.disconnect()

        logging.info("Mapping state complete.")
    except asyncio.CancelledError as ex:
        logging.error("Mapping state canceled")
        raise ex

    # Need to wait for YOLO processing to finish, or we may not have objects to drop at
    if (
        not self.flight_settings.yolo_status.is_set()
        and not self.flight_settings.skip_odlc_and_airdrop
    ):
        logging.info("Waiting for YOLO processing to finish...")
        await self.flight_settings.yolo_status.wait()
        logging.info("YOLO processing finished.")

    return Airdrop(self.drone, self.flight_settings)


# Setting the run_callable attribute of the Mapping class to the run function
Mapping.run_callable = run
