"""
Main driver code for moving drone to each waypoint
"""

import asyncio
import logging
import sys

import dronekit

from flight.waypoint.calculate_distance import calculate_distance

SIM_ADDR: str = "tcp:127.0.0.1:5762"
CON_ADDR: str = "/dev/ttyFTDI"
WAYPOINT_TOLERANCE: int = 6


# Python imports made me angry so I just copied move_to here
async def move_to(
    drone: dronekit.Vehicle, latitude: float, longitude: float, altitude: float
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
        a float contatining the requested altitude to go to (in meters)
    fast_param: float
        a float that determines if the drone will take less time checking its precise location
        before moving on to another waypoint. If its 1, it will move at normal speed,
        if its less than 1(0.83), it will be faster.
    """
    drone.simple_goto(
        dronekit.LocationGlobalRelative(latitude, longitude, altitude),
        airspeed=None,
    )
    location_reached: bool = False

    # First determine if we need to move fast through waypoints or need to slow down at each one
    # Then loops until the waypoint is reached
    logging.info("Going to waypoint")
    while not location_reached:
        position: dronekit.LocationGlobalRelative = drone.location.global_relative_frame

        drone_lat: float = position.latitude_deg
        drone_long: float = position.longitude_deg
        drone_alt: float = position.relative_altitude_m

        total_distance: float = calculate_distance(
            drone_lat,
            drone_long,
            drone_alt,
            latitude,
            longitude,
            altitude,
        )

        if total_distance < WAYPOINT_TOLERANCE:
            location_reached = True
            logging.info("Arrived %sm away from waypoint", total_distance)
            break

        # tell machine to sleep to prevent contstant polling, preventing battery drain
        await asyncio.sleep(1)
    return


# duplicate code disabled for testing function
# pylint: disable=duplicate-code
async def run() -> None:
    """
    Runs
    """

    lats: list[float] = [37.94893290, 37.947899284]
    longs: list[float] = [-91.784668343, -91.782420970]

    # create a drone object
    logging.info("Waiting for drone to connect...")
    drone: dronekit.Vehicle = dronekit.connect(SIM_ADDR, baud=921600, wait_ready=True)

    # initilize drone configurations
    drone.airspeed = 20

    logging.info("Waiting for pre-arm checks to pass...")
    while not drone.is_armable:
        await asyncio.sleep(0.5)

    logging.info("-- Arming")
    drone.mode = dronekit.VehicleMode("GUIDED")
    drone.armed = True
    while drone.mode != "GUIDED" or not drone.armed:
        await asyncio.sleep(0.5)

    logging.info("-- Taking off")
    drone.simple_takeoff(25)

    # wait for drone to take off
    await asyncio.sleep(15)

    # Fly to first waypoint
    print("Going to first waypoint")
    await drone.simple_goto(dronekit.LocationGlobalRelative(lats[0], longs[0], 25))
    await asyncio.sleep(10)

    # Begin 12 mile flight
    print("Starting the line")
    for i in range(43):
        point: int
        for point in range(len(lats)):
            await move_to(drone, lats[point], longs[point], 75)
            print("Reached waypoint")
        print("Iteration:", i)

    # return home
    logging.info("12 miles accomplished")
    logging.info("Returning to home")
    drone.mode = dronekit.VehicleMode("RTL")
    while drone.mode.name != "RTL":
        await asyncio.sleep(0.5)
    while drone.system_status.state != "STANDBY":
        await asyncio.sleep(0.5)

    logging.info("Staying connected...")
    # infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


# Runs through the code until it has looped through each element of
# the Lats and Longs array and the drone has arrived at each of them
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        print("Program ended")
        sys.exit(0)
