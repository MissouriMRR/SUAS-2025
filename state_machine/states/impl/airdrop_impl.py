"""Implements the behavior of the Airdrop state."""

import asyncio
import logging
import json
import math

import utm

from flight.extract_gps import extract_gps
from flight.waypoint.goto import move_to

from state_machine.state_tracker import (
    update_state,
    update_drone,
    update_flight_settings,
)

from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings, SimMode

from state_machine.states.airdrop import Airdrop
from state_machine.states.land import Land
from state_machine.states.state import State
from state_machine.states.waypoint import Waypoint

from vision.common.constants import Location, ODLCDict

# The altitude to go up to while staying at the airdrop point
# This could allow stuck beacons to wiggle out
# 75 ft -> 23 m
WIGGLE_ALTITUDE: float = 23.0


async def run(self: Airdrop) -> State:
    """
    Implements the run method for the Airdrop state.

    Returns
    -------
    Waypoint | Land : State
        Goes to the Waypoint state if there is another airdrop to complete,
        or Land if there are no more airdrops to complete.

    Notes
    -----
    This method is responsible for initiating the Airdrop process of the drone and transitioning
    it back to the Waypoint state.
    """

    if self.flight_settings.skip_odlc_and_airdrop:
        return Land(self.drone, self.flight_settings)

    try:
        update_state("Airdrop")
        update_drone(self.drone)
        update_flight_settings(self.flight_settings)
        logging.info("Airdrop state running")

        with open("flight/data/output.json", encoding="utf8") as output:
            drop_locations: ODLCDict = json.load(output)
            if not drop_locations:
                logging.error("No drop locations found, loading fallback locations")
                fallback_locations: list[Location] = extract_gps(
                    self.flight_settings.mission_data_path
                )["default_airdrop_points"]
                for i, location in enumerate(fallback_locations):
                    drop_locations[f"fallback_{i}"] = location

        with open("flight/data/bottles.json", encoding="utf8") as output:
            cylinders: dict[str, dict[str, int | bool]] = json.load(output)

        logging.info("Moving to drop location")

        # Track attempted locations
        attempted_locations: set[str] = set()
        try:
            with open("flight/data/attempted_drops.json", encoding="utf8") as file:
                attempted_locations = set(json.load(file))
        except (FileNotFoundError, json.JSONDecodeError):
            with open("flight/data/attempted_drops.json", "w", encoding="utf8") as file:
                json.dump(list(attempted_locations), file)

        # Find if there is a loaded cylinder
        cylinder_num: str = ""
        for cylinder in cylinders:
            if cylinders[cylinder]["Loaded"]:
                cylinder_num = cylinder
                break
            if cylinder_num == "":
                cylinder_num = cylinder
        else:
            logging.warning("No beacons are loaded?")
            return Land(self.drone, self.flight_settings)

        dropped: bool = await attempt_drop(
            self.drone,
            self.flight_settings,
            drop_locations,
            cylinders,
            attempted_locations,
            cylinder_num,
            self.flight_settings.mission_data_path,
            self.flight_settings.mean_wind_speed,
            self.flight_settings.mean_wind_direction,
        )

        with open("flight/data/bottles.json", "w", encoding="utf8") as output:
            json.dump(cylinders, output)

        if not dropped:
            return Airdrop(self.drone, self.flight_settings)
        continue_run: bool = False

        for cylinder in cylinders:
            if (cylinders[cylinder])["Loaded"]:
                continue_run = True

        if continue_run:
            return Waypoint(self.drone, self.flight_settings)
        return Land(self.drone, self.flight_settings)

    except asyncio.CancelledError as ex:
        logging.error("Airdrop state canceled")
        raise ex
    finally:
        pass


# pylint: disable=too-many-arguments,too-many-locals
async def attempt_drop(
    drone: Drone,
    flight_settings: FlightSettings,
    drop_locations: ODLCDict,
    cylinders: dict[str, dict[str, int | bool]],
    attempted_locations: set[str],
    cylinder_num: str,
    path: str,
    mean_wind_speed: float,
    mean_wind_direction: float,
    retry_mode: bool = False,
) -> bool:
    """
    Attempts to perform a drop at the next available location.

    Parameters
    ----------
    drone : Drone
        The drone object to control
    flight_settings : FlightSettings
        The flight settings object
    drop_locations : ODLCDict
        Dictionary of available drop locations
    cylinders : dict
        Dictionary of cylinder states
    attempted_locations : set
        Set of previously attempted drop locations
    cylinder_num : str
        The cylinder number to use for the drop
    path : str
        the path to where our waypoint/flight locations are stored
    mean_wind_speed : float, default 0.0
        The mean wind speed, in meters per second.
    mean_wind_direction : float, default 0.0
        The mean wind direction, in degrees.
        A value of 0 represents north, and 90 represents west.
    retry_mode : bool, default False
        Whether we're in retry mode (attempting previously visited locations)

    Returns
    -------
    bool
        True if drop was successful, False otherwise
    """
    try:
        # Find next available drop location
        available_locations = set(drop_locations.keys()) - attempted_locations
        location_id: str
        drop_loc: Location

        if available_locations:
            # Get the next location ID (using min for consistent ordering)
            location_id = min(available_locations)
            drop_loc = drop_locations[location_id]
        else:
            logging.info("All locations attempted, entering retry mode")
            retry_mode = True
            # Pick the first location to retry
            location_id = min(drop_locations.keys())
            drop_loc = drop_locations[location_id]

        airdrop_altitude: float = extract_gps(path)["airdrop_altitude"]

        wind_offset: float = calculate_airdrop_wind_offset(mean_wind_speed, airdrop_altitude)

        easting: float
        northing: float
        zone_number: int
        zone_letter: str
        easting, northing, zone_number, zone_letter = utm.from_latlon(
            drop_loc["latitude"], drop_loc["longitude"]
        )

        easting += wind_offset * -math.sin(math.radians(mean_wind_direction))
        northing += wind_offset * math.cos(math.radians(mean_wind_direction))

        drop_lat: float
        drop_lon: float
        drop_lat, drop_lon = utm.to_latlon(easting, northing, zone_number, zone_letter)

        await move_to(drone.vehicle, drop_lat, drop_lon, airdrop_altitude)

        if retry_mode:
            logging.info(
                "Attempting drop at previously visited location %s",
                location_id,
            )
        else:
            logging.info(
                "Starting drop at fresh location %s",
                location_id,
            )

        if flight_settings.sim_mode is SimMode.REAL:
            await drone.open_servo((cylinders[cylinder_num])["Servo"])

        (cylinders[cylinder_num])["Loaded"] = False

        # Record attempted location
        attempted_locations.add(location_id)
        with open("flight/data/attempted_drops.json", "w", encoding="utf8") as file:
            json.dump(list(attempted_locations), file)

        await asyncio.sleep(9)

        # Attempt to wiggle out stuck beacons
        logging.info("Moving up to 75 ft / 23 meters...")
        await move_to(drone.vehicle, drop_lat, drop_lon, WIGGLE_ALTITUDE)
        await asyncio.sleep(3)

        logging.info("-- Airdrop done!")
        return True

    except KeyError:
        logging.warning("Drop location %s was not found. Skipping.", location_id)
        return False


def calculate_airdrop_wind_offset(wind_speed: float, drop_altitude: float) -> float:
    """
    Calculates the wind offset for dropping the payload.

    Parameters
    ----------
    wind_speed : float
        The wind speed, in meters per second.
    drop_altitude : float
        The altitude of the drop, in meters.

    Returns
    -------
    float
        The offset in the direction of the wind to add to the drop location.
    """
    if wind_speed <= 0.0:
        return 0.0

    payload_mass: float = 0.155  # kg
    gravity_acceleration: float = 9.81  # m/s^2
    parachute_area: float = 0.25  # cross-sectional area in m^2
    parachute_drag_coefficient: float = 1.4  # coefficient of drag
    air_density: float = 1.225  # kg/m^3
    parachute_closed_duration: float = 1.0  # seconds

    vertical_velocity_parachute_open: float = math.sqrt(
        2.0
        * payload_mass
        * gravity_acceleration
        / (air_density * parachute_drag_coefficient * parachute_area)
    )  # m/s

    freefall_distance: float = 0.5 * gravity_acceleration * parachute_closed_duration**2  # meters

    parachute_open_duration: float = (
        drop_altitude - freefall_distance
    ) / vertical_velocity_parachute_open  # seconds
    drop_duration: float = parachute_closed_duration + parachute_open_duration  # seconds

    offset: float = -wind_speed * (
        drop_duration + 0.2 * (math.exp(-5.0 * drop_duration) - 1.0)
    )  # meters

    offset *= 0.9  # should improve accuracy slightly

    return offset


# Setting the run_callable attribute of the Airdrop class to the run function
Airdrop.run_callable = run
