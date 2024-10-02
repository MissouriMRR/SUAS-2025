"""Defines the Drone class for the state machine."""

import dronekit


class Drone:
    """
    A drone for the state machine to control.
    This class is a wrapper around the dronekit Vehicle class,
    and will be passed around to each state.
    Data can be stored in this class to be shared between states.

    Attributes
    ----------
    address : str
        The address used to connect to the drone.
    baud : int | None
        The baud rate, or None to use the default.
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
    vehicle(self) -> dronekit.Vehicle
        Get the Dronekit Vehicle object owned by this Drone object.
    is_connected(self) -> bool
        Checks if a drone has been connected to.
    connect_drone(self) -> Awaitable[None]
        Connect to a drone.
    close(self) -> Awaitable[none]
        Close the owned DroneKit Vehicle object.
    """

    def __init__(self, address: str = "", baud: int | None = None) -> None:
        """
        Initialize a new Drone object, but do not connect to a drone.

        Parameters
        ----------
        address : str, default ""
            The address of the drone to connect to when the `connect_drone()`
            method is called.
        baud : int, default None
            The baud rate, or None to use the default.
        """
        self._vehicle: dronekit.Vehicle | None = None
        self.address: str = address
        self.baud: int | None = baud
        self.odlc_scan: bool = True

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
        """Get the DroneKit Vehicle object owned by this Drone object.

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
            raise RuntimeError("we haven't connected to the drone yet")
        return vehicle

    async def connect_drone(self) -> None:
        """Connect to a drone. This operation is idempotent.

        Raises
        ------
        RuntimeError
            If no connection address has been set.
        """
        if self.is_connected:
            return

        if len(self.address) == 0:
            raise RuntimeError("no connection address specified")

        vehicle: dronekit.Vehicle = (
            dronekit.connect(self.address, wait_ready=True)
            if self.baud is None
            else dronekit.connect(self.address, wait_ready=True, baud=self.baud)
        )

    async def close(self) -> None:
        """Close the owned DroneKit Vehicle object."""
        self.vehicle.close()
