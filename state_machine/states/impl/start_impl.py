"""Implements the behavior of the Start state."""

import asyncio
import logging

import dronekit

from state_machine.state_tracker import update_state
from state_machine.states.start import Start
from state_machine.states.state import State
from state_machine.states.takeoff import Takeoff


async def run(self: Start) -> State:
    """
    Implements the run method for the Start state.

    This method establishes a connection with the drone, waits for the drone to be
    discovered, ensures the drone has a global position estimate, arms the drone,
    and transitions to the Takeoff state.

    Returns
    -------
    Takeoff : State
        The next state after the Start state has successfully run.

    Notes
    -----
    This method is responsible for initializing the drone and transitioning it to the
    Takeoff state, which is the next step in the state machine.

    """
    try:
        update_state("Start")
        logging.info("Start state running")

        # connect to the drone
        logging.info("Waiting for drone to connect...")
        await self.drone.connect_drone()

        logging.info("Waiting for pre-arm checks to pass...")
        while not self.drone.vehicle.is_armable:
            await asyncio.sleep(0.5)

        logging.info("-- Arming")
        self.drone.vehicle.mode = dronekit.VehicleMode("GUIDED")
        self.drone.vehicle.armed = True
        while self.drone.vehicle.mode != "GUIDED" or not self.drone.vehicle.armed:
            await asyncio.sleep(0.5)

        logging.info("Start state complete")
        return Takeoff(self.drone, self.flight_settings)
    except asyncio.CancelledError as ex:
        logging.error("Start state canceled")
        raise ex
    finally:
        pass


# Setting the run_callable attribute of the Start class to the run function
Start.run_callable = run
