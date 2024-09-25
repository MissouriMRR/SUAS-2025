"""Defines the Drone class for the state machine."""

import asyncio

import dronekit


class Drone:
    """
    A drone for the state machine to control.
    This class is a wrapper around the mavsdk System class, and will be passed around to each state.
    Data can be stored in this class to be shared between states.

    Attributes
    ----------
    address : str
        The address used to connect to the drone.
    is_connected
    odlc_scan : bool
        A boolean to tell if the odlc zone needs to be scanned, used the
        first run and if odlc needs to be scanned any other time
    vehicle
    _vehicle : dronekit.Vehicle | None
        The Dronekit Vehicle object that controls the drone, or None if a connection
        hasn't been made yet.

    Methods
    -------
    __init__(connection_string: str) -> None
        Initialize a new Drone object, but do not connect to a drone.
    connect_drone(self) -> Awaitable[None]
        Connect to a drone.
    is_connected(self) -> bool
        Checks if a drone has been connected to.
    vehicle(self) -> dronekit.Vehicle
        Get the Dronekit Vehicle object owned by this Drone object.
    """

    def __init__(self, address: str) -> None:
        """
        Initialize a new Drone object, but do not connect to a drone.

        Parameters
        ----------
        address : str
            The address of the drone to connect to when the `connect_drone()`
            method is called.
        """
        self._vehicle: dronekit.Vehicle | None = None
        self.address: str = address
        self.odlc_scan: bool = True

    async def connect_drone(self) -> None:
        """Connect to a drone. This operation is idempotent."""
        if self.is_connected:
            return

        vehicle: dronekit.Vehicle = dronekit.connect(self.address)
        vehicle.wait_ready(True)  # TODO: modify dronekit and make it async
        self._vehicle = vehicle

    @property
    def is_connected(self) -> bool:
        """Checks if a drone has been connected to.

        Returns
        -------
        bool
            Whether this Drone object has connected to a drone.
        """
        return self._vehicle is not None

    @property
    def vehicle(self) -> dronekit.Vehicle:
        """Get the Dronekit Vehicle object owned by this Drone object.

        Returns
        -------
        dronekit.Vehicle
            The Vehicle object owned by this Drone object.

        Raises
        ------
        AttributeError
            If a connection hasn't been made yet.
        """
        vehicle: dronekit.Vehicle | None = self._vehicle
        if vehicle is None:
            raise AttributeError("we haven't connected to the drone yet")
        return vehicle
