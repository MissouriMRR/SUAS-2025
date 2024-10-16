"""
File for the airdrop unit test
"""

import asyncio
import logging

import dronekit

from state_machine.flight_settings import FlightSettings
from state_machine.state_machine import StateMachine
from state_machine.states import Airdrop
from state_machine.drone import Drone


async def run() -> None:
    """
    Runs the Airdrop unit test
    """

    logging.info("Creating the drone")
    # create a drone object
    drone: Drone = Drone()
    drone.use_sim_settings()
    await drone.connect_drone()

    # initilize drone configurations
    drone.vehicle.airspeed = 20

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
    await asyncio.sleep(10)

    flight_settings: FlightSettings = FlightSettings()

    logging.info("starting airdrop")

    await airdrop_run(drone, flight_settings)

    try:
        # tell machine to sleep to prevent constant polling, preventing battery drain
        await asyncio.sleep(1)

        logging.info("Done!")
    except KeyboardInterrupt:
        logging.critical("Keyboard interrupt detected. Killing state machine and landing drone.")
    finally:
        print("Done")


async def airdrop_run(drone: Drone, flight_settings: FlightSettings) -> None:
    """
    Starts airdrop state of statemachine

    Parameters
    ----------
    drone: Drone
        drone class that includes drone object

    flight_settings: FlightSettings
        settings for flight to be passed into the statemachine
    """
    drone.odlc_scan = False
    await StateMachine(Airdrop(drone, flight_settings), drone, flight_settings).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
