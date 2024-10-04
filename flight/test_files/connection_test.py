"""Does a simple connection test to make sure the computer can connect to the drone."""

import asyncio
import logging
import sys

from state_machine.drone import Drone

SIM_ADDR: str = "tcp:127.0.0.1:5762"  # Address to connect to the simulator
CONTROLLER_ADDR: str = "/dev/ttyFTDI"  # Address to connect to a pixhawk board


async def run_test(sim: bool) -> None:
    """
    Run the state machine.

    Parameters
    ----------
    sim : bool
        Whether to run the state machine in simulation mode.
    """
    address: str = SIM_ADDR if sim else CONTROLLER_ADDR
    drone: Drone = Drone(address=address, baud=921600)
    await drone.connect_drone()

    # connect to the drone
    logging.info("Waiting for drone to connect...")
    while not drone.is_connected:
        await asyncio.sleep(1)

    logging.info("Drone discovered!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test("--sim" in sys.argv))
