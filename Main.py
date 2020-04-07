"""main.py
==========================================================================================
Initialises viewer object and retrieves required control values dependent on flight phase.

Harry Rance 07/04/2020."""

import sys
sys.path.append('/usr/local/lib')

import cv2
import Viewer
## Initialisation of viewer object.
v = Viewer.Viewer()

while True:
    ## Tick count 1 for calculating framerate of system.
    e1 = cv2.getTickCount()
    v.update()
    ## Tick count 2 for calculating framerate of system.
    e2 = cv2.getTickCount()
    ## Total time taken for function to execute.
    time1 = (e2 - e1) / cv2.getTickFrequency()
    ## Control command to be sent to drone.
    control = v.get_drone_control()
    ## Flight phase of the drone.
    phase = v.deployment_phase

    if phase:
        print("########################################")
        print("In deployment Phase? {}".format(v.deployment_phase))
        print("CONTROL: {} ".format(control))
        print("Deployment Area Distances: X = {} mm     Y = {} mm".format(round(v.deployment_area_distances_mm[0], 3),
                                                                          round(v.deployment_area_distances_mm[1], 3)))
    else:
        print("########################################")
        print("In deployment Phase? {}".format(v.deployment_phase))
        print("CONTROL: {} ".format(control))
        print("Distance To Pillar: {} mm".format(round(v.distance_from_pillar, 3)))
    v.show_window()

    #print("Time Update: {},".format(time1))

