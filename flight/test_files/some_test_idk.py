import json
from typing import Optional
from typing import Dict, Any

def update_state(file_path: str, new_state: str) -> None:
    """
    Updates the 'state' field in a JSON file.
    
    :param file_path: Path to the JSON file to be updated.
    :param new_state: New state value as a string to replace the current state.
    """
    # Step 1: Read the current data from the file
    try:
        with open(file_path, 'r') as file:
            data: Dict[str, Any] = json.load(file)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {file_path}.")
        return
    
    # Step 2: Update the 'state' field with the new state
    data['state'] = new_state
    
    # Step 3: Write the updated dictionary back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"State updated to '{new_state}' in {file_path}.")

def read_state_from_json(file_path: str) -> Optional[str]:
    """
    Reads the 'state' field from a JSON file and returns it.
    
    :param file_path: Path to the JSON file to be read.
    :return: The value of the 'state' field or None if the file cannot be read or the field is missing.
    """
    try:
        with open(file_path, 'r') as file:
            data: Dict[str, Any] = json.load(file)
            return data.get('state', None)  # Returns None if 'state' key is not found
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {file_path}.")
    
    return None  # Return None if there was an error reading the file

def update_drone(file_path: str, address: str, odlc_scan: bool, num: int) -> None:
    """
    Updates values in the 'drone' category of a JSON file.
    
    :param file_path: Path to the JSON file to be updated.
    :param address: New address value (str).
    :param odlc_scan: New odlc_scan value (bool).
    :param servo_num: New servo_num value (int).
    :param num: New num value (int).
    """
    try:
        with open(file_path, 'r') as file:
            data: Dict[str, Any] = json.load(file)
        
        if 'drone' not in data:
            print("No 'drone' category in JSON.")
            return
        
        if address is not None:
            data['drone']['address'] = address
        if odlc_scan is not None:
            data['drone']['odlc_scan'] = odlc_scan
        if num is not None:
            data['drone']['num'] = num
        
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        print("Drone data updated successfully.")
        
    except Exception as e:
        print(f"Error updating drone data: {e}")

def update_flight_settings(file_path: str, simple_takeoff: bool, title: str, description: str, waypoint_count: int) -> None:
    """
    Updates values in the 'flight_settings' category of a JSON file.
    
    :param file_path: Path to the JSON file to be updated.
    :param simple_takeoff: New simple_takeoff value (bool).
    :param title: New title value (str).
    :param description: New description value (str).
    :param waypoint_count: New waypoint_count value (int).
    """
    try:
        with open(file_path, 'r') as file:
            data: Dict[str, Any] = json.load(file)
        
        if 'flight_settings' not in data:
            print("No 'flight_settings' category in JSON.")
            return
        
        if simple_takeoff is not None:
            data['flight_settings']['simple_takeoff'] = simple_takeoff
        if title is not None:
            data['flight_settings']['title'] = title
        if description is not None:
            data['flight_settings']['description'] = description
        if waypoint_count is not None:
            data['flight_settings']['waypoint_count'] = waypoint_count
        
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        print("Flight settings updated successfully.")
        
    except Exception as e:
        print(f"Error updating flight settings: {e}")
