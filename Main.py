import sys
sys.path.append('/usr/local/lib')

import pyrealsense2 as rs
import numpy as np
import cv2
import Viewer
import Autopilot

v = Viewer.Viewer()
#a = Autopilot.Autopilot()

def pos_lr():
    lines = v.get_lr_boundaries()

    if len(lines) == 2:
        line_l, line_r = lines

        # Generate Equation of lines in form a0x + b0y = c0 (1) & a1x + b1y = c1 (2)
        al, bl, cl, ar, br, cr = v.get_line_eq(line_l, line_r)

        # To find X point using specified Y, (c-by)/a = x
        y = 240

        x_l = (cl - (bl * y)) / al
        x_r = (cr - (br * y)) / ar

        centre = (x_l + x_r) / 2

        d_centre = centre - (848/2)

        move_dir = ' '

        if d_centre < 0.:
            move_dir = 'LEFT'
        elif d_centre > 0.:
            move_dir = 'RIGHT'
        else:
            move_dir = 'NONE'

        #print("Movement Direction: {}".format(move_dir))

        return move_dir

while True:
    e1 = cv2.getTickCount()
    v.update()
    e2 = cv2.getTickCount()
    time1 = (e2 - e1) / cv2.getTickFrequency()
    control = v.get_drone_control()

    #print("########################################")
    #print("     CONTROL: {} ".format(control))
    #print("In deployment Phase? {}".format(v.deployment_phase))
    print("Deployment Area Distances: {}".format(v.deployment_area_distances_mm))
    v.show_window()

    #print("Time Update: {},".format(time1))


    #LR_move_direction = pos_lr()
