"""Implements the behavior of the Land state."""

import asyncio
import logging

import dronekit

from state_machine.state_tracker import (
    update_state,
    update_drone,
    update_flight_settings,
)
from state_machine.states.land import Land


async def run(self: Land) -> None:
    """
    Implements the run method for the Land state.

    This method initiates the landing process of the drone and transitions to the Start state.

    Returns
    -------
    Start : State
        The next state after the drone has successfully landed.

    Notes
    -----
    This method is responsible for initiating the landing process of the drone and transitioning
    it back to the Start state, preparing for a new flight.

    """
    try:
        update_state("Land")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)

        logging.info("Landing")

        # Instruct the drone to land
        self.drone.vehicle.mode = dronekit.VehicleMode("RTL")
        while self.drone.vehicle.mode.name != "RTL":
            await asyncio.sleep(0.5)

        while self.drone.vehicle.system_status.state != "STANDBY":
            await asyncio.sleep(0.5)

        while self.drone.vehicle.armed:
            await asyncio.sleep(0.5)

        logging.info("Land state complete.")
        return
    except asyncio.CancelledError as ex:
        logging.error("Land state canceled")
        raise ex


# Setting the run_callable attribute of the Land class to the run function
Land.run_callable = run
