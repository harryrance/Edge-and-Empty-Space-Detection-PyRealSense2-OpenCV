from dronekit import connect, VehicleMode, LocationGlobalRelative, mavutil
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

vehicle = connect('/dev/cu.usbmodem1411', wait_ready=True, baud=115200)
print("!!! Connection to the vehicle on: {} !!!".format(connection_string))


#Define function for take off
def arm_and_takeoff(secs):
    print("!!! Arming Motors !!!")
    '''
    while not vehicle.is_armable:
        print("!!! Waiting for vehicle to initialise... !!!")
        time.sleep(1)
    '''
    vehicle.mode = VehicleMode("STABILIZE")
    vehicle.armed = True

    '''
    while not vehicle.armed:
        print("!!! Waiting for arming... !!!")
        time.sleep(1)
    '''
    print("!!! Takeoff !!!")

    time.sleep(1)

def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    print("###### 1 ######")
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 1 Hz cycle
    for x in range(0, duration):
        print("###### 2 ######")
        vehicle.send_mavlink(msg)
        time.sleep(1)



# Main
arm_and_takeoff(10)
#send_ned_velocity(3,0,0,5)
#time.sleep(1)
send_ned_velocity(30, 30, 30, 15)
time.sleep(1)
print("###### 3 ######")


vehicle.mode = VehicleMode('LAND')

# Close Connection
vehicle.close()

