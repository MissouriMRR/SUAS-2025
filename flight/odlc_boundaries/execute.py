"""
File containing the move_to function responsible
for moving the drone to a certain waypoint
"""

import asyncio
import logging
from mavsdk import System


# duplicate code disabled since we may want different functionality
# for waypoints/odlcs search points
# pylint: disable=duplicate-code
async def move_to(
    drone: System,
    latitude: float,
    longitude: float,
    altitude: float,
) -> None:
    """
    This function takes in a latitude, longitude and altitude and autonomously
    moves the drone to that waypoint. This function will also auto convert the altitude
    from feet to meters.

    Parameters
    ----------
    drone: System
        a drone object that has all offboard data needed for computation
    latitude: float
        a float containing the requested latitude to move to
    longitude: float
        a float containing the requested longitude to move to
    altitude: float
        a float containing the requested altitude to go to (in feet)
    """

    # converts feet into meters
    altitude_in_meters: float = altitude * 0.3048

    # get current altitude
    async for terrain_info in drone.telemetry.home():
        absolute_altitude: float = terrain_info.absolute_altitude_m
        break

    await drone.action.goto_location(latitude, longitude, altitude_in_meters + absolute_altitude, 0)
    location_reached: bool = False
    # First determine if we need to move fast through waypoints or need to slow down at each one
    # Then loops until the waypoint is reached
    while not location_reached:
        logging.info("Going to waypoint")
        async for position in drone.telemetry.position():
            # continuously checks current latitude, longitude and altitude of the drone
            drone_lat: float = position.latitude_deg
            drone_long: float = position.longitude_deg
            drone_alt: float = position.relative_altitude_m

            # roughly checks if location is reached and moves on if so
            if (
                (round(drone_lat, 4) == round(latitude, 4))
                and (round(drone_long, 4) == round(longitude, 4))
                and (round(drone_alt, 1) == round(altitude_in_meters, 1))
            ):
                location_reached = True
                logging.info("arrived")
                break

        # tell machine to sleep to prevent contstant polling, preventing battery drain
        await asyncio.sleep(1)
    return
