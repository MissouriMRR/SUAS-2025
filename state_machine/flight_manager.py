"""Tests the state machine."""

import asyncio
import logging
from mavsdk.telemetry import FlightMode, LandedState

from state_machine.drone import Drone
from state_machine.state_machine import StateMachine
from state_machine.states import Start
from state_machine.flight_settings import DEFAULT_STANDARD_OBJECT_COUNT, FlightSettings


class FlightManager:
    """
    Class that manages the state machine, kill switch, and gracefully exiting the program.

    Methods
    -------
    __init__(self) -> None
        Initialize a flight manager object.
    run_manager() -> Awaitable[None]
        Run the state machine until completion in a separate process.
        Sets the drone address to the simulation or physical address.
    _run_state_machine(drone: Drone) -> None
        Create and run a state machine until completion in the event loop.
        This method should be called in its own process.
    run_kill_switch(state_machine_process: Process, drone: Drone) -> None
        Create and run a kill switch in the event loop.
    _kill_switch(state_machine_process: Process, drone: Drone) -> Awaitable[None]
        Enable the kill switch and wait until it activates. The drone should be
        in manual mode after this method returns.
    _graceful_exit(drone: Drone) -> Awaitable[None]
        Lands the drone and exits the program.
    """

    def __init__(self) -> None:
        self.drone: Drone = Drone()

    async def run_manager(
        self,
        sim_flag: bool,
        path_data_path: str = "flight/data/waypoint_data.json",
        skip_waypoint: bool = False,
        standard_object_count: int = DEFAULT_STANDARD_OBJECT_COUNT,
    ) -> None:
        """
        Run the state machine until completion in a separate process.
        Sets the drone address to the simulation or physical address.

        Parameters
        ----------
        sim_flag : bool
            A flag representing if the drone is a simulation.
        path_data_path : str, default "flight/data/waypoint_data.json"
            The path to the JSON file containing the boundary and waypoint data.
        skip_waypoint : bool, default False
            Whether to skip the waypoint state.
        standard_object_count : int, default DEFAULT_STANDARD_OBJECT_COUNT
            The number of standard objects to attempt to find.
        """
        if sim_flag:
            self.drone.address = "udp://:14540"
        else:
            self.drone.address = "serial:///dev/ttyFTDI:921600"

        flight_settings_obj: FlightSettings = FlightSettings(
            sim_flag=sim_flag,
            path_data_path=path_data_path,
            skip_waypoint=skip_waypoint,
            standard_object_count=standard_object_count,
        )
        logging.info("Initializing drone connection")
        await self.drone.connect_drone()

        logging.info("Starting processes")

        state_machine_task: asyncio.Task[None] = asyncio.ensure_future(
            StateMachine(
                Start(self.drone, flight_settings_obj), self.drone, flight_settings_obj
            ).run()
        )

        asyncio.ensure_future(self.kill_switch(state_machine_task))

        try:
            while not state_machine_task.done():
                await asyncio.sleep(0.25)

            logging.info("State machine task has completed. Exiting...")
            return
        except KeyboardInterrupt:
            logging.critical(
                "Keyboard interrupt detected. Killing state machine and landing drone."
            )
            state_machine_task.cancel()
            await self._graceful_exit()

    async def kill_switch(self, state_machine_process: asyncio.Task[None]) -> None:
        """
        Enable the kill switch and wait until it activates. The drone should be
        Continuously check for whether or not the kill switch has been activated.
        in manual mode after this method returns.

        Parameters
        ----------
        state_machine_process: asyncio.Task
            The task running the state machine to kill. This task will
            be cancelled.
        """

        # connect to the drone
        logging.debug("Kill switch running")

        async for connection_state in self.drone.system.core.connection_state():
            if connection_state.is_connected:
                logging.info("Kill switch has been enabled.")
                break

        async for flight_mode in self.drone.system.telemetry.flight_mode():
            if flight_mode == FlightMode.POSCTL:
                break
            await asyncio.sleep(0.5)

        logging.critical("Kill switch activated. Terminating state machine.")

        state_machine_process.cancel()

    async def _graceful_exit(self) -> None:
        """
        Land the drone and exit the program.
        """
        await self.drone.connect_drone()
        logging.critical("Beginning graceful exit. Landing drone...")
        await self.drone.system.action.return_to_launch()
        async for state in self.drone.system.telemetry.landed_state():
            if state == LandedState.ON_GROUND:
                logging.info("Drone landed successfully.")
                break
        logging.info("Drone landed. Exiting program...")
        return
