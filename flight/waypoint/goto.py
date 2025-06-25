"""
File containing the move_to function responsible
for moving the drone to a certain waypoint and stopping there for 15 secs
"""

import asyncio
import logging

import dronekit

from flight.waypoint.calculate_distance import calculate_distance

# Waypoint tolerance in meters: 6 meters = 19.685 feet
WAYPOINT_TOLERANCE: int = 12


# duplicate code disabled since we may want different functionality
# for waypoints/odlcs search points
# pylint: disable=duplicate-code
async def move_to(
    drone: dronekit.Vehicle,
    latitude: float,
    longitude: float,
    altitude: float,
    airspeed: float | None = None,
    tolerance: float | None = None,
) -> None:
    """
    This function takes in a latitude, longitude and altitude and autonomously
    moves the drone to that waypoint.

    Parameters
    ----------
    drone : dronekit.Vehicle
        The drone to move.
    latitude : float
        The requested latitude to move to, in degrees.
    longitude : float
        The requested longitude to move to, in degrees.
    altitude : float
        The requested altitude to go to, in meters.
    airspeed : float, default None
        The requested airspeed in meters per second,
        or None to let DroneKit decide the airspeed.
    tolerance : float, default None
        The tolerance in meters, or None to use the default tolerance of 6 meters.
    """
    if tolerance is None:
        tolerance = 6

    drone.simple_goto(
        dronekit.LocationGlobalRelative(latitude, longitude, altitude),
        airspeed=airspeed,
    )
    location_reached: bool = False
    # First determine if we need to move fast through waypoints or need to slow down at each one
    # Then loops until the waypoint is reached
    logging.info("Going to waypoint")
    while not location_reached:
        position: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

        # continuously checks current latitude, longitude and altitude of the drone
        drone_lat: float = position.lat
        drone_long: float = position.lon
        drone_alt: float = position.alt

        total_distance: float = calculate_distance(
            drone_lat,
            drone_long,
            drone_alt,
            latitude,
            longitude,
            altitude,
        )

        if total_distance < tolerance:
            location_reached = True
            logging.info("Arrived %sm away from waypoint", total_distance)
            break

        # tell machine to sleep to prevent constant polling, preventing battery drain
        await asyncio.sleep(1.0)
    return
