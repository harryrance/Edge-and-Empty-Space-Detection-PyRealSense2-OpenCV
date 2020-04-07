from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import dronekit_sitl


class Autopilot:
    def __init__(self):
        sitl = dronekit_sitl.start_default()
        connection_string = sitl.connection_string()

        vehicle = connect(connection_string, wait_ready=True)
        print("!!! Connection to the vehicle on: {} !!!".format(connection_string))