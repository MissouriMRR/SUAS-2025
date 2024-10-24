"""
File containing the move_to function responsible
for moving the drone to a certain waypoint and stopping there for 15 secs
"""

import asyncio
import logging

from mavsdk import System

from flight.waypoint.calculate_distance import calculate_distance

# Waypoint tolerance in meters: 6 meters = 19.685 feet
WAYPOINT_TOLERANCE: int = 6


# duplicate code disabled since we may want different functionality
# for waypoints/odlcs search points
# pylint: disable=duplicate-code
async def move_to(drone: System, latitude: float, longitude: float, altitude: float) -> None:
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
        a float containing the requested altitude to go to in meters
    """

    # get current altitude
    async for terrain_info in drone.telemetry.home():
        absolute_altitude: float = terrain_info.absolute_altitude_m
        break

    await drone.action.goto_location(latitude, longitude, altitude + absolute_altitude, 0)
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

            total_distance: float = calculate_distance(
                drone_lat, drone_long, drone_alt, latitude, longitude, altitude
            )

            if total_distance < WAYPOINT_TOLERANCE:  # 6 meters = 19.685 feet.
                location_reached = True
                logging.info("Arrived %sm away from waypoint", total_distance)
                break

        # tell machine to sleep to prevent constant polling, preventing battery drain
        await asyncio.sleep(1)
    return
