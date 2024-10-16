"""Runs the state machine and kill switch in separate processes in order to test them."""

import asyncio
from state_machine.flight_manager import FlightManager


async def run_test(sim: bool) -> None:
    """
    Run the state machine.

    Parameters
    ----------
    sim : bool
        Whether to run the state machine in simulation mode.
    """
    await FlightManager().run_manager(sim)


if __name__ == "__main__":
    asyncio.run(run_test(True))
