"""Defines the Drone class for the state machine."""

import mavsdk


class Drone:
    """
    A drone for the state machine to control.
    This class is a wrapper around the mavsdk System class, and will be passed around to each state.
    Data can be stored in this class to be shared between states.

    Attributes
    ----------
    system : mavsdk.System
        The MAVSDK system object that controls the drone.
    address : str
        The address used to connect to the drone when the `connect_drone()`
        method is called.

    Methods
    -------
    __init__(address: str) -> None
        Initialize a new Drone object, but do not connect to a drone.
    connect_drone(self) -> Awaitable[None]
        Connect to a drone.
    """

    def __init__(self, address: str = "udp://:14540") -> None:
        """
        Initialize a new Drone object, but do not connect to a drone.

        Parameters
        ----------
        address : str
            The address of the drone to connect to when the `connect_drone()`
            method is called.
        """
        self.system = mavsdk.System()
        self.address = address

    async def connect_drone(self) -> None:
        """Connect to a drone."""
        await self.system.connect(system_address=self.address)