"""Implements the behavior of the Land state."""

import asyncio
import logging

from mavsdk.action import ActionError
from mavsdk.telemetry import FlightMode
from state_machine.state_tracker import update_state, update_drone, update_flight_settings
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
        await self.drone.system.action.return_to_launch()

        await asyncio.sleep(5)
        async for flight_mode in self.drone.system.telemetry.flight_mode():
            if flight_mode != FlightMode.RETURN_TO_LAUNCH:
                break
            await asyncio.sleep(1)

        try:
            await self.drone.system.action.disarm()
        except ActionError:
            logging.warning("Unable to disarm, may already be disarmed.")

        logging.info("Land state complete.")
        return
    except asyncio.CancelledError as ex:
        logging.error("Land state canceled")
        raise ex


# Setting the run_callable attribute of the Land class to the run function
Land.run_callable = run
