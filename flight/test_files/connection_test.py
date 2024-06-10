"""Does a simple connection test to make sure the computer can connect to the drone."""

import asyncio
import logging
import sys

from state_machine.drone import Drone

SIM_ADDR: str = "udp://:14540"  # Address to connect to the simulator
CONTROLLER_ADDR: str = "serial:///dev/ttyFTDI"  # Address to connect to a pixhawk board


async def run_test(sim: bool) -> None:
    """
    Run the state machine.

    Parameters
    ----------
    sim : bool
        Whether to run the state machine in simulation mode.
    """
    address: str = SIM_ADDR if sim else CONTROLLER_ADDR
    drone: Drone = Drone(address)
    await drone.connect_drone()

    # connect to the drone
    logging.info("Waiting for drone to connect...")
    async for state in drone.system.core.connection_state():
        if state.is_connected:
            logging.info("Drone discovered!")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test("--sim" in sys.argv))
