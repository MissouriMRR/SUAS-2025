"""
Used to update State and flight settings in json with kill switch
"""

import json
import logging
from typing import Any

from state_machine.drone import Drone
from state_machine.flight_settings import FlightSettings

DEFAULT_STATE_PATH = "state_machine/state.json"


def read_state_data(file_path: str = DEFAULT_STATE_PATH) -> None | dict[str, Any]:
    """
    Reads the data from the state JSON file.

    Parameters
    ----------
    file_path : str, optional
        The file path of the JSON file, by default "state_machine/state.json"

    Returns
    -------
    None|Dict[str, Any]
        The data from the JSON file, or None if there was an error reading the file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.warning("File %s not found.", file_path)
    except json.JSONDecodeError:
        logging.warning("Error decoding JSON from file %s.", file_path)

    return None  # Return None if there was an error reading the file


def update_state(new_state: str, file_path: str = DEFAULT_STATE_PATH) -> None:
    """
    Updates the 'state' field in the JSON file with the new state.

    Parameters
    ----------
    new_state : str
        The new state to update the 'state' field with.
    file_path : str, optional
        The file path of the JSON file, by default "state_machine/state.json"
    """
    # Step 1: Read the current data from the file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)
    except FileNotFoundError:
        logging.warning("File %s not found.", file_path)
        return
    except json.JSONDecodeError:
        logging.warning("Error decoding JSON from file %s.", file_path)
        return

    # Step 2: Update the 'state' field with the new state
    data["state"] = new_state

    # Step 3: Write the updated dictionary back to the JSON file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    logging.info("State updated to '%s' in %s.", new_state, file_path)


def read_state_from_json(file_path: str = DEFAULT_STATE_PATH) -> None | str:
    """
    Reads and returns the 'state' field from a JSON file.

    Parameters
    ----------
    file_path : str, optional
        The file path of the JSON file, by default "state_machine/state.json"

    Returns
    -------
    None|str
        The 'state' field from the JSON file, or None if there was an error reading the file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)
            return data.get("state", None)  # Returns None if 'state' key is not found
    except FileNotFoundError:
        logging.warning("File %s not found.", file_path)
    except json.JSONDecodeError:
        logging.warning("Error decoding JSON from file %s.", file_path)

    return None  # Return None if there was an error reading the file


def update_drone(drone: Drone, file_path: str = DEFAULT_STATE_PATH) -> None:
    """
    Updates the drone-specific information in the 'drone' category of a JSON file.

    Parameters
    ----------
    drone : Drone
        The drone object containing the updated information.
    file_path : str, optional
        The file path of the JSON file, by default "state_machine/state.json"
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)

        if "drone" not in data:
            logging.warning("No 'drone' category in JSON.")
            return

        if drone.address is not None:
            data["drone"]["address"] = drone.address
        if drone.odlc_scan is not None:
            data["drone"]["odlc_scan"] = drone.odlc_scan

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        logging.info("Drone data updated successfully.")

    except FileNotFoundError:
        logging.warning("File %s not found.", file_path)
        return
    except json.JSONDecodeError:
        logging.warning("Error decoding JSON from file %s.", file_path)
        return


def update_flight_settings(
    flight_settings: FlightSettings, file_path: str = DEFAULT_STATE_PATH
) -> None:
    """
    Updates the flight settings in the 'flight_settings' category of a JSON file.

    Parameters
    ----------
    flight_settings : FlightSettings
        The flight settings object containing the updated information.
    file_path : str, optional
        The file path of the JSON file, by default "state_machine/state.json"
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)

        if "flight_settings" not in data:
            logging.warning("No 'flight_settings' category in JSON.")
            return

        if flight_settings.simple_takeoff is not None:
            data["flight_settings"]["simple_takeoff"] = flight_settings.simple_takeoff
        if flight_settings.run_title is not None:
            data["flight_settings"]["title"] = flight_settings.run_title
        if flight_settings.run_description is not None:
            data["flight_settings"]["description"] = flight_settings.run_description

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        logging.info("Flight settings updated successfully.")

    except FileNotFoundError:
        logging.warning("File %s not found.", file_path)
        return
    except json.JSONDecodeError:
        logging.warning("Error decoding JSON from file %s.", file_path)
        return
