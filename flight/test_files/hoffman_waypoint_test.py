"""
Main driver code for moving drone to each waypoint
"""

import asyncio
import logging
import sys

import dronekit

from flight import extract_gps
from flight.waypoint.goto import move_to
from state_machine.drone import Drone

# Defining file path constant for extract_gps
MOVE_TO_TEST_PATH: str = "./data/waypoint_data.json"

# Defining altitude and speed
MOVE_TO_TEST_ALTITUDE: int = 12
MOVE_TO_TEST_SPEED: int = 20


async def run() -> None:
    """
    This function is a driver to test the goto function and runs through the
    given waypoints in the lats and longs lists at the altitude of 100.
    Makes the drone move to each location in the lats and longs arrays at the altitude of 100.

    Notes
    -----
    Currently has 3 values in each the Lats and Longs array and code is looped
    and will stay in that loop until the drone has reached each of locations
    specified by the latitude and longitude and
    continues to run until forced disconnect
    """

    # Put all latitudes, longitudes and altitudes into separate arrays
    lats: list[float] = []
    longs: list[float] = []
    altitudes: list[float] = []

    waypoint_data = extract_gps.extract_gps(MOVE_TO_TEST_PATH)
    waypoints = waypoint_data["waypoints"]

    waypoint: tuple[float, float, float]
    for waypoint in waypoints:
        lats.append(waypoint.latitude)
        longs.append(waypoint.longitude)
        altitudes.append(waypoint.altitude)

    # create a drone object
    drone: Drone = Drone()
    drone.use_sim_settings()
    await drone.connect_drone()

    # initilize drone configurations
    drone.vehicle.airspeed = MOVE_TO_TEST_SPEED

    # connect to the drone
    logging.info("Waiting for pre-arm checks to pass...")
    while not drone.vehicle.is_armable:
        await asyncio.sleep(0.5)

    logging.info("-- Arming")
    drone.vehicle.mode = dronekit.VehicleMode("GUIDED")
    drone.vehicle.armed = True
    while drone.vehicle.mode.name != "GUIDED" or not drone.vehicle.armed:
        await asyncio.sleep(0.5)

    logging.info("-- Taking off")
    drone.vehicle.simple_takeoff(MOVE_TO_TEST_ALTITUDE)

    # wait for drone to take off
    while drone.vehicle.location.global_relative_frame.alt < MOVE_TO_TEST_ALTITUDE - 0.25:
        await asyncio.sleep(1)

    # wait for drone to take off
    await asyncio.sleep(10)

    # move to each waypoint in mission
    point: int
    for point in range(len(lats)):
        await move_to(drone, lats[point], longs[point], 100)

    # return home
    logging.info("Last waypoint reached")
    logging.info("Returning to home")
    drone.vehicle.mode = dronekit.VehicleMode("RTL")
    while drone.vehicle.mode.name != "RTL":
        await asyncio.sleep(0.5)
    while drone.vehicle.system_status.state != "STANDBY":
        await asyncio.sleep(0.5)
    print("Staying connected, press Ctrl-C to exit")

    # infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


# Runs through the code until it has looped through each element of
#  the Lats and Longs array and the drone has arrived at each of them
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        logging.info("CTRL+C: Program ended")
        sys.exit(0)
