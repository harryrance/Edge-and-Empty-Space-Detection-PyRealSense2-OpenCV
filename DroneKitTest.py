from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

# Connect to vehicle
import argparse
parser = argparse.ArgumentParser(description='commands')
parser.add_argument('--connect')
args = parser.parse_args()

connection_string = args.connect

sitl = None


if not connection_string:
    import dronekit_sitl
    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()

vehicle = connect(connection_string, wait_ready=True)
print("!!! Connection to the vehicle on: {} !!!".format(connection_string))


#Define function for take off
def arm_and_takeoff(tgt_altitude):
    print("!!! Arming Motors !!!")

    while not vehicle.is_armable:
        print("!!! Waiting for vehicle to initialise... !!!")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True


    while not vehicle.armed:
        print("!!! Waiting for arming... !!!")
        time.sleep(1)

    print("!!! Takeoff !!!")

    vehicle.simple_takeoff(tgt_altitude)

    # Wait to reach target altitude
    while True:
        altitude = vehicle.location.global_relative_frame.alt

        #altitude += 5

        if altitude >= tgt_altitude - 1:
            print("Altitude Reached: {}".format(altitude))
            break


        time.sleep(1)

# Main
arm_and_takeoff(10)

# Set Default Speed
vehicle.airspeed = 3

# Go to waypoint
print("Go To Waypoint 1")
wp1 = LocationGlobalRelative(35.9872609, -80.8753037, 20)

vehicle.simple_goto(wp1)

# Wait
time.sleep(30)

# Come Home
print("Return Home")
vehicle.mode = VehicleMode("RTL")

time.sleep(20)

# Close Connection
vehicle.close()

