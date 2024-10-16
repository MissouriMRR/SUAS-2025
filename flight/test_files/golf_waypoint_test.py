"""
Main driver code for moving drone to each waypoint
"""

import asyncio
import logging
import sys

from dronekit import VehicleMode

from flight.waypoint.goto import move_to
from state_machine.drone import Drone

WAYPOINT_TOLERANCE: int = 6  #


# duplicate code disabled for testing function
# pylint: disable=duplicate-code
async def run(sim: bool) -> None:
    """
    This function is a driver to test the goto function and runs through the
    given waypoints in the lats and longs lists at the altitude of 100.
    Makes the drone move to each location in the lats and longs arrays
    at the altitude of 100 and

    Notes
    -----
    Currently has 4 values in each the Lats and Longs array and code is looped
    and will stay in that loop until the drone has reached each of locations
    specified by the latitude and longitude and continues to run until forced disconnect
    """
    # Put all latitudes, longitudes and altitudes into separate arrays
    lats: list[float] = [37.948658, 37.948200, 37.948358, 37.948800]
    longs: list[float] = [-91.784431, -91.783406, -91.783253, -91.784169]

    # create a drone object
    drone: Drone = Drone()
    if sim:
        drone.use_sim_settings()
    else:
        drone.use_real_settings()

    logging.info("Waiting for drone to connect...")
    await drone.connect_drone()
    logging.info("Drone discovered!")

    # initilize drone configurations
    drone.vehicle.airspeed = 10

    # connect to the drone
    logging.info("Waiting for pre-arm checks to pass...")
    while not drone.vehicle.is_armable:
        await asyncio.sleep(0.5)

    logging.info("-- Arming")
    drone.vehicle.mode = VehicleMode("GUIDED")
    drone.vehicle.armed = True
    while drone.vehicle.mode.name != "GUIDED" or not drone.vehicle.armed:
        await asyncio.sleep(0.5)

    logging.info("-- Taking off")
    drone.vehicle.simple_takeoff(15)

    # wait for drone to take off
    while drone.vehicle.location.global_relative_frame.alt < 14.9:
        await asyncio.sleep(1)

    # move to each waypoint in mission
    for i in range(2):
        logging.info("Starting loop %s", i)
        for point in range(len(lats)):
            await move_to(drone.vehicle, lats[point], longs[point], 15)

    # return home
    logging.info("Last waypoint reached")
    logging.info("Returning to home")
    drone.vehicle.mode = VehicleMode("RTL")
    while drone.vehicle.mode.name != "RTL":
        await asyncio.sleep(0.5)
    while drone.vehicle.system_status.state != "STANDBY":
        await asyncio.sleep(0.5)
    print("Staying connected, press Ctrl-C to exit")

    # infinite loop till forced disconnect
    while True:
        await asyncio.sleep(1)


# Runs through the code until it has looped through each element of
# the Lats and Longs array and the drone has arrived at each of them
if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run("--sim" in sys.argv))
    except KeyboardInterrupt:
        print("Program ended")
        sys.exit(0)
