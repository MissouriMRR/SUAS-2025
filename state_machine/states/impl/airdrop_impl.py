"""Implements the behavior of the Airdrop state."""

import asyncio
import logging
import json

from state_machine.state_tracker import update_state

from state_machine.states.airdrop import Airdrop
from state_machine.states.land import Land
from state_machine.states.waypoint import Waypoint
from state_machine.states.state import State

# uncomment if automatic
# from flight.maestro.air_drop import AirdropControl
from flight.waypoint.goto import move_to


async def run(self: Airdrop) -> State:
    """
    Implements the run method for the Airdrop state.

    Returns
    -------
    Waypoint : State
        The next state after the drone has successfully completed the Airdrop.

    Notes
    -----
    This method is responsible for initiating the Airdrop process of the drone and transitioning
    it back to the Waypoint state.
    """
    try:
        update_state("Airdrop")
        logging.info("Airdrop")
        # uncomment if automatic
        # if self.drone.address == "serial:///dev/ttyFTDI:921600":
        #   setup airdrop
        #   airdrop = AirdropControl()

        # if automatic un comment servo num
        bottle: int
        # servo_num: int
        cylinder_num: str

        with open("flight/data/output.json", encoding="utf8") as output:
            bottle_locations = json.load(output)

        with open("flight/data/bottles.json", encoding="utf8") as output:
            cylinders = json.load(output)

        logging.info("Moving to bottle drop")

        # setting a priority for bottles
        if (cylinders["C1"])["Loaded"]:
            bottle = (cylinders["C1"])["Bottle"]
            # servo_num = (cylinders["C1"])["Bottle"]
            cylinder_num = "C1"
        elif (cylinders["C2"])["Loaded"]:
            bottle = (cylinders["C2"])["Bottle"]
            # servo_num = (cylinders["C2"])["Bottle"]
            cylinder_num = "C2"
        else:
            logging.warning("No bottles are loaded?")
            return Land(self.drone, self.flight_settings)

        dropped: bool = False

        try:
            bottle_loc: dict[str, float] = bottle_locations[str(bottle)]

            # Move to the bottle with priority
            await move_to(self.drone.system, bottle_loc["latitude"], bottle_loc["longitude"], 24)
            logging.info(
                "Starting bottle drop %s. Wait for drone to be stationary then drop.", bottle
            )
            # If bottle drop is automatic these would be used
            # if self.drone.address == "serial:///dev/ttyFTDI:921600":
            #   await airdrop.drop_bottle(servo_num)

            dropped = True
            (cylinders[cylinder_num])["Loaded"] = False

            await asyncio.sleep(
                15
            )  # This will need to be changed based on how long it takes to drop the bottle

            logging.info("-- Airdrop done!")
        except KeyError:
            # This means the location for the bottle loaded wasn't found.
            logging.warning("Info for bottle %s was not found. Skipping.", bottle)
            (cylinders[cylinder_num])["Loaded"] = False
            dropped = False

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


# Setting the run_callable attribute of the Airdrop class to the run function
Airdrop.run_callable = run
