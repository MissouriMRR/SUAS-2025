"""
File for test way point path for SUAS 3 miles in length
"""

import asyncio
import logging
import sys

import dronekit

from flight.waypoint.goto import move_to
from state_machine.drone import Drone


async def run() -> None:
    """
    run simple waypoint flight path
    """

    # create a drone object
    drone: Drone = Drone()
    drone.use_sim_settings()
    await drone.connect_drone()

    # initilize drone configurations
    drone.vehicle.airspeed = 30

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
    drone.vehicle.simple_takeoff(12)

    # wait for drone to take off
    while drone.vehicle.location.global_relative_frame.alt < 11.9:
        await asyncio.sleep(1)

    # wait for drone to take off
    await asyncio.sleep(60)

    accuracy_mod: float = 0.83
    obj_altitude: float = 12
    points: list[tuple[float, float]] = [
        (38.31413, -76.54352),
        (38.31629, -76.55587),
        (38.31611, -76.55126),
        (38.31712, -76.55102),
        (38.31560, -76.54838),
        (38.31413, -76.54352),
        (38.31629, -76.55587),
        (38.31413, -76.54352),
        (38.31466, -76.54665),
    ]

    point: tuple[float, float]
    for point in points:
        await move_to(drone.vehicle, point[0], point[1], obj_altitude, accuracy_mod)

    # return home
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
# the Lats and Longs array and the drone has arrived at each of them
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        print("Program ended")
        sys.exit(0)
