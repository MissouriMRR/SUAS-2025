"""Implements the behavior of the Takeoff state."""

import asyncio
import logging

from flight.extract_gps import extract_gps
from state_machine.state_tracker import update_state
from state_machine.states.state import State
from state_machine.states.takeoff import Takeoff
from state_machine.states.waypoint import Waypoint


async def run(self: Takeoff) -> State:
    """
    Implements the run method for the Takeoff state.

    This method initiates the drone takeoff process and transitions to the Waypoint state.

    Returns
    -------
    Waypoint : State
        The next state after a successful takeoff.

    Raises
    ------
    asyncio.CancelledError
        If the execution of the Takeoff state is canceled.

    Notes
    -----
    This method is responsible for taking off the drone and transitioning it to the
    Waypoint state, which represents the navigation phase to reach a specified waypoint.

    """
    try:
        update_state("Takeoff")
        logging.info("Takeoff state running")

        # Set takeoff altitude to the minimum allowed altitude, plus one meter
        # 3.28084 feet per meter
        takeoff_altitude: float = (
            extract_gps(self.flight_settings.path_data_path)["altitude_limits"][0] / 3.28084 + 1.0
        )
        logging.info("Using takeoff altitude of %f m", takeoff_altitude)

        self.drone.vehicle.simple_takeoff(takeoff_altitude)

        # Wait until the drone has stopped taking off
        while True:
            altitude: float = self.drone.vehicle.location.global_relative_frame.alt
            logging.info(
                "Current altitude: %f",
                altitude,
            )
            if altitude >= takeoff_altitude - 0.5:
                logging.info("Reached target altitude")
                break
            await asyncio.sleep(1.0)

        return Waypoint(self.drone, self.flight_settings)
    except asyncio.CancelledError as ex:
        logging.error("Takeoff state canceled")
        raise ex
    finally:
        pass


# Setting the run_callable attribute of the Takeoff class to the run function
Takeoff.run_callable = run
