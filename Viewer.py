"""viewer.py
==============================================================================================================
Core module of the project - processes all image and depth data to provide required outputs for drone control.

Harry Rance 07/04/2020."""

import sys

sys.path.append('/usr/local/lib')

import pyrealsense2 as rs
import numpy as np
import cv2
import operator
import math
import time

from statistics import mean


class Viewer:
    """Class for the Realsense Viewer"""
    def __init__(self):
        """Contructor Function."""
        ## PyRealsense Object Initialiser.
        self.pipeline = rs.pipeline()
        config = rs.config()

        ## Global integer for screen width (default=848 pixels)
        self.screen_width = 848
        ## Global integer for screen height (default=480 pixels)
        self.screen_height = 480
        config.enable_stream(rs.stream.depth, self.screen_width, self.screen_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.screen_width, self.screen_height, rs.format.bgr8, 30)

        ## Initialises the profile for the depth and colour streams.
        self.profile = self.pipeline.start(config)

        align_to = rs.stream.color
        ## Alignment object to align the depth stream with colour stream
        self.align = rs.align(align_to)

        ## Initialise colouriser for depth stream.
        self.colouriser = rs.colorizer()
        ## Initialise decimation filter object.
        self.decimation = rs.decimation_filter()
        ## Initialise spatial filter object.
        self.spatial = rs.spatial_filter()
        ## Initialise temporal filter object.
        self.temporal = rs.temporal_filter()
        ## Initialise hole filling filter object.
        self.hole_filling = rs.hole_filling_filter()
        ## Transform depth stream to disparity map.
        self.depth_to_disparity = rs.disparity_transform(True)
        ## Transform disparity map to depth stream.
        self.disparity_to_depth = rs.disparity_transform(False)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        ## Initialise depth scale object for converting raw distances into different units (default=millimetres.)
        self.depth_scale = depth_sensor.get_depth_scale()
        ## List containing all of the active contours, used to verify whether or not the contours are usable.
        self.active_contours = []
        clipping_dist_m = 5
        ## Initialise clipping distance with use of depth scale. Used to eliminate background from image.
        self.clipping_dist = clipping_dist_m / self.depth_scale
        ## Boolean - used to identify whether or not the deployment area is safe and deployable.
        self.verified_deployment_area = False
        ## Boolean - used to identify whether or not the drone is in the deployment phase.
        self.deployment_phase = False

        ## Dict containing key: movement direction, value: movement distance (pixels). For flight phase.
        self.pre_deploy_dir = {}
        ## String containing directions to move.
        self.direction_to_move = ''
        ## Float value containing distance to move.
        self.distance_to_move = 0
        ## Float value containing distance from 'pillar'.
        self.distance_from_pillar = 0.

        ## List containing tuples of deployment area coordinates.
        self.deployment_area_coordinates = []
        ## Dict containing key: movement direction, value: movement distance (millimetres). For deployment phase.
        self.deployment_dir_to_move = {}

        ## Boolean - identifies if the internal contour is active.
        self.internal_contour = False
        ## Boolean - identifies if the external contour is active.
        self.external_contour = False

        ## Float value containing scale factor to shrink deployment area by.
        self.contour_shrink_sf = 1.

        ## Tuple containing coordinates of top left corner of deployment area.
        self.deploy_tl = (0, 0)
        ## Tuple containing coordinates of top right corner of deployment area.
        self.deploy_tr = (0, 0)
        ## Tuple containing coordinates of bottom left corner of deployment area.
        self.deploy_bl = (0, 0)
        ## Tuple containing coordinates of bottom right corner of deployment area.
        self.deploy_br = (0, 0)

        ## List containing the width and height of deployment area in millimetres.
        self.deployment_area_distances_mm = []

    def adapt_depth_clipping(self, depth_image):
        """Function to implement adaptive depth clipping.
        Takes in the nearest object's depth and eliminated background behind this with a 20% tolerance.

        Parameters
        ----------
            depth_image:
                The image data of the depth stream from the Realsense camera.
        Returns
        -------
            closest_object_tol:
                The distance of the closes object (+20%) tolerance in metres."""
        e1 = cv2.getTickCount()
        # Define X and Y bounds to check depth in (search middle third). Keep as round numbers so that steps of 2, 5 or 10 can be used
        _xmin = 0
        _ymin = 130

        _xmax = 840
        _ymax = 250

        # Initialise closest object value
        depth_arr = []

        # Scan every 10th pixel for depth, and exclude 0.00 values
        for _x in range(_xmin, _xmax, 20):
            for _y in range(_ymin, _ymax, 20):
                depth = depth_image.get_distance(_x, _y)
                if depth != 0.:
                    depth_arr.append(depth)

        # Find closest stored distance
        closest_obj = min(depth_arr)

        # Add 20% for tolerance
        closest_obj_tol = closest_obj * 1.2

        e2 = cv2.getTickCount()
        time1 = (e2 - e1) / cv2.getTickFrequency()
        return closest_obj_tol

    def get_move_dir(self, lr_dir):
        """ Function to get the movement direction in the drone's flight phase.
        Takes input directions and outputs readable control commands.

        Parameters
        ----------
            lr_dir:
                Move left, right or centred command received from the get_x_dist() function.
        Returns
        -------
            self.pre_deploy_dir:
                Dictionary containing key: direction for the drone to move, value: number of pixels away from centre."""

        drone_dir = ' '
        #print("LR DIR: {}".format(lr_dir))
        if lr_dir is not None:
            if lr_dir[0] == 'move left':
                drone_dir = 'L'
            elif lr_dir[0] == 'move right':
                drone_dir = 'R'
            elif lr_dir[0] == 'centered':
                drone_dir = 'X'
            else:
                pass

            self.pre_deploy_dir = {drone_dir : lr_dir[1]}

    def get_drone_control(self):
        """ Function to get the drone control commands, dependent on the flight phase - either flying, or deploying.
        Outputs the direction in which the drone needs to move, dependent on the phase of flight.

        Returns
        -------
            self.pre_deploy_dir:
                Dictionary containing key: direction for the drone to move, value: number of pixels away from centre.

            self.deployment_dir_to_move:
                Deployment direction if in deployment phase."""

        if not self.deployment_phase:
            return self.pre_deploy_dir
        else:
            return self.deployment_dir_to_move

    # Detect edges in the image [CHECK THRESHOLDS]
    def canny_edge(self, image):
        """ Function to process the colour image.
        Takes in colour image data and outputs image edge data, in a BW thresholded format.

        Parameters
        ----------
            image:
               The colour image data to be processed.
        Returns
        -------
            edges:
                The edge data to be used in subsequent processing functions."""

        # Convert image to greyscale format from BG
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Remove noise from image (larger h value removes more noise, but loses image detail)
        #image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        image = cv2.blur(image, (5,5))
        image = cv2.bilateralFilter(image, 9,150,150)
        # Threshold the image to find edges and convert to binary image
        edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 11, 2)

        # Invert image from black on white to white on black, so that Houghlines can detect edges
        edges = cv2.bitwise_not(edges)

        # Dilate edges to allow for more lines to be generated to give a larger sample size
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges
    # Function used to find the orientation of the target with respect to the camera's coordinate space
    def check_z_orientation(self, centre):
        """ Function to check the orientation about the Z axis of the object being viewed.
        Outputs a visual of the orientation of the object relative to the horizontal.

        Parameters
        ----------
            centre:
                Centre point of the contour of the object."""

        # Pass in the centre point of the object, and create a left and right point with respect to an x offset
        offset = 30
        right = (centre[0] + offset, centre[1])
        left = (centre[0] - offset, centre[1])

        # Check to make sure that the selected pixels are active in the depth frame
        if self.get_colour(int(centre[0]), int(centre[1])) != (0, 0, 0):
            if self.get_colour(int(right[0]), int(right[1])) != (0, 0, 0):

                # Get the distance in metres to each point
                l_depth = self.aligned_depth_frame.get_distance(int(left[0]), int(left[1]))
                r_depth = self.aligned_depth_frame.get_distance(int(right[0]), int(right[1]))
                c_depth = self.aligned_depth_frame.get_distance(int(centre[0]), int(centre[1]))

                # Use the intrinsic depth data to deproject the pixel into its positional coordinates (in m) relative
                # to an arbitrary coordinate space
                c_intrin = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(centre[0]), int(centre[1])], c_depth)
                r_intrin = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(right[0]), int(right[1])], r_depth)

                # Find the distances in the x-direction and z-direction. Pythagoras can then be used on a triangle
                # as if the target was being viewed from a birds-eye perspective
                x_dist = r_intrin[0] - c_intrin[0]
                z_dist = r_intrin[2] - c_intrin[2]

                # Calculate the angle of the target with respect to the angle of the camera
                theta_r = math.atan((z_dist/x_dist))
                theta_d = math.degrees(theta_r)

                x1, y1 = 700, 100

                length = 100

                x2 = int(x1 + length * math.cos(theta_r))
                y2 = int(y1 + length * math.sin(theta_r))

                # Draw a graphical representation on the screen of the angle of the target
                cv2.line(self.bg_removed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(self.bg_removed, (x1, y1), (x1 + 100, y1), (0, 0, 255), 2)

    def get_dist_to_pillar(self, rect):
        """ Get the distance between the Realsense camera and the 'bridge support pillar'.
        Outputs the distance to the target in millimetres.

        Parameters
        ----------
            rect:
                List containing the centre point of the contour, the height and width of the contour, and the angle of rotation through the y-axis (into/out of the screen).
        Returns
        -------
            dist:
                Distance to the viewed object in millimetres."""

        midpoint, _, _ = rect

        mid_x, mid_y = midpoint

        dist = self.aligned_depth_frame.get_distance(int(mid_x), int(mid_y))/self.depth_scale

        return dist

    def find_contours(self, edges):
        """ Function used to find the contours of the object being viewed. Contains operations to calculate the outer
        contour and inner contour, and also draws the contours on the screen, and processes contour data to output
        multiple variables, including the deployment phase and some movement directions.

        Parameters
        ----------
            edges:
               The edge data from the canny_edge() function.
        Returns
        -------
            box:
                Box object containing coordinates of the four corners of the bounding contour box.
            rect:
                List containing the centre point of the contour, the height and width of the contour, and the angle of rotation through the y-axis (into/out of the screen).
            contours_int:
                Contours object containing all internal contour data."""

        self.verified_deployment_area = False
        # Find the contours of the viewed object. 'cv2.RETR_EXTERNAL' specifies that only the external contour is to be used.
        im = cv2.cvtColor(self.bg_removed, cv2.COLOR_BGR2GRAY)
        #contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # The following code ensures that there will always be a contour drawn on the screen.
        # 1 - 10 contours are stored. the last contour's area is examined. If it is within a threshold, it is ignored.
        # 2 - If a contour's perimeter is outside the set threshold, the areas array is iterated through to find an
        #     acceptable area, then the corresponding contour is used instead.

        # Initialise contours, perimeter and area arrays
        self.active_contours.append(contours)

        storage_length = 10
        perims = []
        areas = []

        # Only proceed if there are 10 contours stored
        if len(self.active_contours) == storage_length:
            # Append perims and areas into respective arrays
            for i in range(0, storage_length):
                perims.append(cv2.arcLength(self.active_contours[i][0], True))
                areas.append(cv2.contourArea(self.active_contours[i][0]))

            # Find the current area and average of last 10 areas
            avg_area = mean(areas)
            curr_area = cv2.contourArea(self.active_contours[i][0])

            self.internal_contour = False
            self.external_contour = False

            if len(areas) >= storage_length:
                # If the current area is greater and average, or within 15% less than average, use the most recently
                # stored contour
                if (curr_area > avg_area) or ((avg_area - (avg_area*0.15)) < curr_area):
                    int_cnt, ext_cnt = self.separate_contours(self.active_contours[-1], hierarchy)
                    #contours = self.active_contours[-1]
                    contours = ext_cnt
                    self.external_contour = True
                    if int_cnt:
                        square_gap = max(int_cnt, key=len)

                        contours_int = int_cnt

                        int_perim = cv2.arcLength(square_gap, True)

                        if int_perim > 80.:

                            self.internal_contour = True
                            self.deployment_phase = True

                        else:
                            self.internal_contour = False
                            self.deployment_phase = False
                    else:
                        square_gap = [0, 0, 0, 0]
                        contours_int = [[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,]]
                        self.internal_contour = False
                        self.deployment_phase = False
                    #print("Int Cnt If: {}".format(int_cnt))
                    #Remove the oldest contour from the list
                    del(self.active_contours[0])

                else:
                    # If the contour area is outside the set threshold, scan the last 10 contours to find a suitable one
                    for j in range(0, storage_length):
                        if areas[storage_length - 1 - j] > avg_area or ((avg_area - (avg_area*0.15)) < areas[storage_length - 1 - j]):
                            # If a contour area is within the threshold, use this contour.
                            int_cnt, ext_cnt = self.separate_contours(self.active_contours[storage_length - 1 - j], hierarchy)
                            #contours = self.active_contours[storage_length - 1 - j]
                            contours = ext_cnt
                            self.external_contour = True
                            if int_cnt:
                                square_gap = max(int_cnt, key=len)

                                contours_int = int_cnt

                                int_perim = cv2.arcLength(square_gap, True)

                                if int_perim > 80.:
                                    self.internal_contour = True
                                    self.deployment_phase = True
                                else:
                                    self.internal_contour = False
                                    self.deployment_phase = False
                            else:
                                square_gap = [0, 0, 0, 0]
                                contours_int = [[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,]]
                                self.internal_contour = False
                                self.deployment_phase = False
                            #print("Int Cnt Else: {}".format(int_cnt))
                            del (self.active_contours[0])
                            break
                        else:
                            square_gap = [0, 0, 0, 0]
                            contours_int = [[0, 0, 0, 0, ], [0, 0, 0, 0, ], [0, 0, 0, 0, ], [0, 0, 0, 0, ]]
            else:
                square_gap = [0, 0, 0, 0]
                contours_int = [[0, 0, 0, 0, ], [0, 0, 0, 0, ], [0, 0, 0, 0, ], [0, 0, 0, 0, ]]
            # Use the first set of contours
            #print('Len: {} '.format(len(contours)))
            if len(contours) > 0:
                cnt = contours[0]

                # Calculate an approximation of the contour to reduce any error.
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Draw the contours and a vertical bounding rectangle
                cv2.drawContours(self.bg_removed, [approx], -1, (0, 0, 255), 5)
                #cv2.drawContours(self.bg_removed, contours_int, -1, (0, 255, 0), 5)
                if square_gap != [0, 0, 0, 0]:
                    cv2.drawContours(self.bg_removed, square_gap, -1, (255, 255, 0), 5)

                # Calculate the minimum area rectangle (mid point, width, height, rotation) and convert the values into an
                # OpenCV box vector. Use this vector to draw a rotated bounding box that will always be a perfect rectangle.
                rect = cv2.minAreaRect(cnt)

                box = cv2.boxPoints(rect)
                box = np.int0(box)

                if not self.deployment_phase:

                    self.distance_from_pillar = self.get_dist_to_pillar(rect)
                else:

                    self.distance_from_pillar = 'NA'

                mid, w_h, theta = rect
                move_dir_lr = self.get_x_dist(mid, w_h, theta)
                self.check_z_orientation(mid)
                self.get_move_dir(move_dir_lr)

                cv2.drawContours(self.bg_removed, [box], 0, (0,255,0), 2)

                if self.internal_contour:

                    int_cnt_perim = cv2.arcLength(square_gap, True)
                    self.deployment_phase = False

                    if int_cnt_perim > 80.:
                        self.deployment_phase = True
                        M = cv2.moments(square_gap)
                        c_x = int(M["m10"] / M["m00"])
                        c_y = int(M["m01"] / M["m00"])

                        cnt_norm = square_gap - [c_x, c_y]

                        cnt_scaled = cnt_norm * self.contour_shrink_sf
                        cnt_scaled = cnt_scaled + [c_x, c_y]
                        cnt_scaled = cnt_scaled.astype(np.int32)
                        scaled_rect = cv2.minAreaRect(cnt_scaled)

                        scaled_box = cv2.boxPoints(scaled_rect)

                        #print(scaled_rect, scaled_box)
                        scaled_box = np.int0(scaled_box)

                        if not self.verified_deployment_area:
                            self.deploy_tl, self.deploy_tr, self.deploy_bl, self.deploy_br = \
                                self.scale_internal_contour(scaled_rect[0], scaled_rect[1])

                        cv2.circle(self.bg_removed, (c_x, c_y), 2, (255, 255, 255), -1)

                        gap_rect = cv2.minAreaRect(square_gap)

                        gap_box = cv2.boxPoints(gap_rect)
                        gap_box = np.int0(gap_box)

                        cv2.drawContours(self.bg_removed, [gap_box], 0, (255, 0, 255), 2)

                        if self.verified_deployment_area:

                            cv2.drawContours(self.bg_removed, [scaled_box], 0, (0, 255, 255), 2)
                            self.get_deployment_coords(self.deploy_tl, self.deploy_tr, self.deploy_bl, self.deploy_br, gap_rect, rect)

                            self.get_deployment_dir_to_move(c_x, c_y)



                return box, rect, contours_int

        false_return = [[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,]]

        return 0, 0, false_return

    def get_deployment_dir_to_move(self, c_x, c_y):
        """ Gets the direction for the drone to move in the deployment phase.
        Outputs a Dict object with {direction to move : coordinates of movement}

        Parameters
        ----------
            c_x:
                Centre point x coordinate of the internal (gap) contour.
            c_y:
                Centre point y coordinate of the internal (gap) contour."""

        s_c_x, s_c_y = (int(self.screen_width/2), int(self.screen_height/2))

        c_x, c_y = int(c_x), int(c_y)

        dx = s_c_x - c_x
        dy = s_c_y - c_y

        dir_x = dir_y = 0

        if dx > 0:
            dir_x = 'L'
        if dx < 0:
            dir_x = 'R'
        if dx == 0:
            dir_x = '_'

        if dy > 0:
            dir_y = 'U'
        if dy < 0:
            dir_y = 'D'
        if dy == 0:
            dir_y = '_'
        self.deployment_dir_to_move = {}

        final_dir = dir_x + dir_y
        coords = (dx, dy)

        self.deployment_dir_to_move = {final_dir : coords}
        #print(self.deployment_dir_to_move)

    def get_deployment_coords(self, tl, tr, bl, br, gap_rect, full_rect):
        """ Gets the coordinates of the internal contour in the deployment phase.
        Shrinks the contour to output verified coordinates so that the drone does not attempt to deploy directly against one of the dap's edges.

        Parameters
        ----------
            tl:
                Original top-left coordinate of contour.
            tr:
                Original top-right coordinate of contour.
            bl:
                Original bottom-left coordinate of contour.
            br:
                Original bottom-right coordinate of contour.
            gap_rect:
                The rect data (midpoint, width, height and theta) of the gap contour.
            full_rect:
                The rect data (midpoint, width, height and theta) of the main object contour.

        Returns
        -------
            self.deployment_area_coordinates:
                List of verified coordinates for the drone to deploy the ground robot within."""

        self.deploy_tl = tl
        self.deploy_tr = tr
        self.deploy_bl = bl
        self.deploy_br = br

        self.deployment_area_coordinates = [self.deploy_tl, self.deploy_tr, self.deploy_bl, self.deploy_br]

        gap_mid, gap_wh, _ = gap_rect
        full_mid, full_wh, _ = full_rect
        #print("Gap Mid: {} Full Mid: {} ".format(gap_mid, full_mid))
        #print("Gap WH: {} Full WH: {}".format(gap_wh, full_wh))
        gap_w, gap_h = gap_wh
        full_w, full_h = full_wh

        gap_ty = gap_mid[1] - gap_h/2
        gap_by = gap_mid[1] + gap_h/2
        gap_lx = gap_mid[0] - gap_w/2
        gap_rx = gap_mid[0] + gap_w/2

        full_ty = full_mid[1] - full_h/2
        full_by = full_mid[1] + full_h/2
        full_lx = full_mid[0] - full_w/2
        full_rx = full_mid[0] + full_w/2

        avg_ty = math.sqrt(math.pow((gap_ty + full_ty), 2)) / 2
        avg_by = math.sqrt(math.pow((gap_by + full_by), 2)) / 2
        avg_lx = math.sqrt(math.pow((gap_lx + full_lx), 2)) / 2
        avg_rx = math.sqrt(math.pow((gap_rx + full_rx), 2)) / 2

        #print("Averages: {} {} {} {}".format(avg_ty, avg_by, avg_lx, avg_rx))

        #print("Deployment Coords: {}".format(self.deployment_area_coordinates))
        #self.deployment_area_distances_mm[0].append(self.deployment_area_coordinates)
        shifted_top_1, shifted_top_2 = self.shift_line_y_t(tl, tr, int(avg_ty))
        d_top_mm = self.deproj_to_pt(shifted_top_1, shifted_top_2)

        shifted_btm_1, shifted_btm_2 = self.shift_line_y_b(bl, br, int(avg_by))
        d_btm_mm = self.deproj_to_pt(shifted_btm_1, shifted_btm_2)

        shifted_left_1, shifted_left_2 = self.shift_line_x_l(tl, bl, int(avg_lx))
        d_left_mm = self.deproj_to_pt(shifted_left_1, shifted_left_2)

        shifted_right_1, shifted_right_2 = self.shift_line_x_r(tr, br, int(avg_rx))
        d_right_mm = self.deproj_to_pt(shifted_right_1, shifted_right_2)

        avg_x_dist = (d_top_mm[0] + d_btm_mm[0]) / 2
        avg_y_dist = (d_left_mm[1] + d_right_mm[1]) / 2

        self.deployment_area_distances_mm = [avg_x_dist, avg_y_dist]

        #print(tl, tr, bl, br)
        #print("Top: {}  Bottom: {}  Left: {}    Right: {}".format(d_top_mm, d_btm_mm, d_left_mm, d_right_mm))

        return self.deployment_area_coordinates

    def scale_internal_contour(self, centre, dims):
        """ Function used to verify the points of the internal contour for deployment.
        Scales contour with respect to whether or not it is verified as a proper area of deployment.

        Parameters
        ----------
            centre:
                Centre point of internal contour.
            dims:
                Width and height of internal contour.

        Returns
        -------
            tl:
                Scaled top-left coordinate.
            tr:
                Scaled top-right coordinate.
            bl:
                Scaled bottom-left coordinate.
            br:
                Scaled bottom-right coordinate"""

        c_x, c_y = centre
        c_x, c_y = int(c_x), int(c_y)

        t_w, t_h = dims

        w, h = int(t_w / 2), int(t_h / 2)

        tl = (c_x - w, c_y - h)
        tr = (c_x + w, c_y - h)

        bl = (c_x - w, c_y + h)
        br = (c_x + w, c_y + h)

        points = [tl, bl, tr, br]

        self.verify_deployment_area(points)

        if self.verified_deployment_area == False:
            self.contour_shrink_sf -= 0.05

        return tl, tr, bl, br

    def verify_deployment_area(self, shrunk_points):
        """ Function to verify whether the edges of the internal contour cross any point of the surrounding support
        structure. If the edges do cross the structure, the drone cannot deploy as it will collide with pillar.
        If they do not, then there is a verified empty space for the drone to deploy the ground robot within.

        Parameters
        ----------
            shrunk_points:
                The shrunken inner contour of the gap."""

        tl, bl, tr, br = shrunk_points

        no_collisions = True

        tl_x, tl_y = tl
        tr_x, tr_y = tr
        bl_x, bl_y = bl
        br_x, br_y = br

        #First, check along top edge
        for x in range(int(tl_x), int(tr_x)):
            #for y in range(int(tl_y), int(tr_y)):
            #print("TOP: {}".format(self.get_colour(x, tl_y)))
            if self.get_colour(x, tl_y) == (0, 0, 0):
                no_collisions = True
            else:
                no_collisions = False

        # Bottom Edge
        for x in range(int(bl_x), int(br_x)):
            #for y in range(int(bl_y), int(br_y)):
            #print("BTM: {}".format(self.get_colour(x, bl_y)))
            if self.get_colour(x, bl_y) == (0, 0, 0):
                no_collisions = True
            else:
                no_collisions = False

        # Left Edge
        for y in range(int(tl_y), int(bl_y)):
            #for x in range(int(tl_x), int(bl_x)):
            #print("L: {}".format(self.get_colour(tl_x, y)))
            if self.get_colour(tl_x, y) == (0, 0, 0):
                no_collisions = True
            else:
                no_collisions = False

        # Right Edge
        for y in range(int(tr_y), int(br_y)):
            #for x in range(int(tr_x), int(br_x)):
            #print("R: {}".format(self.get_colour(tr_x, y)))
            if self.get_colour(tr_x, y) == (0, 0, 0):
                no_collisions = True
            else:
                no_collisions = False

        if no_collisions == True:
            self.verified_deployment_area = True
        else:
            self.verified_deployment_area = False

    def separate_contours(self, contours, hierarchy):
        """ Takes in the full contour list, alongside heirarchy and separates these into internal and external contours.
        Outputs independedt internal and external contour lists.

        Parameters
        ----------
            contours:
                The full list of all detected contours in the colour image.
            hierarchy:
                The hierarchy of the contours: i.e. If they are internal (having parent contours) or external
                      (having no parent contours and being the initial of the group).

        Returns
        -------
            int_cnt:
                Internal contour list.
            ext_cnt:
                External contour list."""

        # If hierarchy[0][i][3] == -1, the contour is external. Else, internal
        int_cnt = []
        ext_cnt = []
        #if self.internal_contour and self.external_contour:
        for i in range(0, len(hierarchy[0])):
            if hierarchy[0][i][3] == -1 and i < len(contours):
                ext_cnt.append(contours[i])
            if hierarchy[0][i][3] > -1 and i < len(contours):
                int_cnt.append(contours[i])

        if int_cnt:
            self.internal_contour = True
            self.deployment_phase = True
        if ext_cnt:
            self.external_contour = True

        #print("INT: {} EXT: {}".format(int_cnt, ext_cnt))
        return int_cnt, ext_cnt

    def draw_roi(self, box, theta):
        """ Draws a region of interest (ROI) above the pillar if no 'gap' is detected.
        Essentially detects the top edge of the pillar.

        Parameters
        ----------
            box:
                The external contour box coordinates
            theta:
                The angle of rotation of the box through the y-axis.

        Returns
        -------
            [t_l_, t_r_, p_l, p_r, t_l, t_r, b_l, b_r]:
                A list of the deployable coordinates, given that the ROI is a verified deployment area."""

        b_l = box[0]
        t_l = box[1]
        t_r = box[2]
        b_r = box[3]

        all_y = [box[0][1], box[1][1], box[2][1], box[3][1]]
        all_x = [box[0][0], box[1][0], box[2][0], box[3][0]]

        index_1, _max = min(enumerate(all_y), key=operator.itemgetter(1))

        maxy_1 = all_y.pop(index_1)
        maxx_1 = all_x.pop(index_1)

        ind1 = (maxx_1, maxy_1)

        index_2, _max = min(enumerate(all_y), key=operator.itemgetter(1))

        maxy_2 = all_y.pop(index_2)
        maxx_2 = all_x.pop(index_2)

        ind2 = (maxx_2, maxy_2)

        if ind2[0] >= ind1[0]:
            t_l = ind1
            t_r = ind2
        else:
            t_l = ind2
            t_r = ind1

        #print("#############################\n1: {} \n2: {} \n3: {} \n4: {} ".format(b_l, t_l, t_r, b_r))

        #Get equation of top line in form a0x +b0y = c0 where c = y intercept and m = slope
        a0, b0, c0, a1, b1, c1 = self.get_line_eq((t_l, t_r), (b_l, b_r))
        if a0 != 0:
            perp_slope = -(1/a0)

            #y = mx + c - find perpendicular y intercept by subbing in points and new slope
            c0_perp_l = (perp_slope * t_l[0]) + t_l[1]
            c0_perp_r = (perp_slope * t_r[0]) + t_r[1]

            #find equations of both perpendicular lines
            line_left = ((t_l), (0, c0_perp_l))
            line_right = ((t_r), (0, c0_perp_r))

            p_a0, p_b0, p_c0, p_a1, p_b1, p_c1 = self.get_line_eq(line_left, line_right)

            #Find point on perpendicular lines which is 30 pixels away in y-dir
            p_ly = t_l[1] - 80
            p_ry = t_r[1] - 80

            t_l_ = (int(t_l[0]), int(t_l[1]))
            t_r_ = (int(t_r[0]), int(t_r[1]))
            if p_a0 != 0 and p_a1 != 0:
                p_lx = (p_c0 - p_ly)/p_a0
                p_rx = (p_c1 - p_ry) / p_a1

                p_l = (int(p_lx), int(p_ly))
                p_r = (int(p_rx), int(p_ry))

                #cv2.line(self.bg_removed, (t_l_), (p_l), (255, 0, 0), 2)
                #cv2.line(self.bg_removed, (t_r_), (p_r), (255, 0, 0), 2)
                #cv2.line(self.bg_removed, (p_l), (p_r), (255, 0, 0), 2)

                return t_l_, t_r_, p_l, p_r, t_l, t_r, b_l, b_r
        return 0,0,0,0,0,0,0,0

    def shift_line_y_t(self, point1, point2, avg_ty):
        """ Shifts the top line of the internal gap contour in the nevative y direction.
        Outputs a set of shifted coordinates.

        Parameters
        ----------
            point1:
                The first point of the internal gap contour top line.
            point2:
                The second point of the internal gap contour top line.

        Returns
        -------
            (pt1_x, pt1_y):
                Tuple containing x and y coordinates of shifted point 1.
            (pt2_x, pt2_y):
                Tuple containing x and y coordinates of shifted point 2."""

        pt1_colour = self.get_colour(point1[0], point1[1])
        pt2_colour = self.get_colour(point2[0], point2[1])

        pt1_x, pt2_x = point1[0], point2[0]
        pt1_y, pt2_y = point1[1], point2[1]
        #print("Original Coords: {}, {}".format(point1, point2))
        if pt1_colour == (0, 0, 0) or pt2_colour == (0, 0, 0):
            pt1_y = avg_ty
            pt2_y = avg_ty

            #print("Shifted Coords: {}, {}".format((pt1_x, pt1_y), (pt2_x, pt2_y)))
            return (pt1_x, pt1_y), (pt2_x, pt2_y)
        else:
            return (pt1_x, pt1_y), (pt2_x, pt2_y)

    def shift_line_y_b(self, point1, point2, avg_by):
        """ Shifts the bottom line of the internal gap contour in the positive y direction.
        Outputs a set of shifted coordinates.

        Parameters
        ----------
            point1:
                The first point of the internal gap contour bottom line.
            point2:
                The second point of the internal gap contour bottom line.

        Returns
        -------
            (pt1_x, pt1_y):
                Tuple containing x and y coordinates of shifted point 1.
            (pt2_x, pt2_y):
                Tuple containing x and y coordinates of shifted point 2."""

        pt1_colour = self.get_colour(point1[0], point1[1])
        pt2_colour = self.get_colour(point2[0], point2[1])

        pt1_x, pt2_x = point1[0], point2[0]
        pt1_y, pt2_y = point1[1], point2[1]
        #print("Original Coords: {}, {}".format(point1, point2))
        if pt1_colour == (0, 0, 0) or pt2_colour == (0, 0, 0):
            pt1_y = avg_by
            pt2_y = avg_by

            #print("Shifted Coords: {}, {}".format((pt1_x, pt1_y), (pt2_x, pt2_y)))
            return (pt1_x, pt1_y), (pt2_x, pt2_y)
        else:
            return (pt1_x, pt1_y), (pt2_x, pt2_y)

    def shift_line_x_r(self, point1, point2, avg_rx):
        """ Shifts the right hand line of the internal gap contour in the positive x direction.
        Outputs a set of shifted coordinates.

        Parameters
        ----------
            point1:
                The first point of the internal gap contour right hand line.
            point2:
                The second point of the internal gap contour right hand line.

        Returns
        -------
            (pt1_x, pt1_y):
                Tuple containing x and y coordinates of shifted point 1.
            (pt2_x, pt2_y):
                Tuple containing x and y coordinates of shifted point 2."""

        pt1_colour = self.get_colour(point1[0], point1[1])
        pt2_colour = self.get_colour(point2[0], point2[1])

        pt1_x, pt2_x = point1[0], point2[0]
        pt1_y, pt2_y = point1[1], point2[1]
        #print("Original Coords: {}, {}".format(point1, point2))
        if pt1_colour == (0, 0, 0) or pt2_colour == (0, 0, 0):
            pt1_x = avg_rx
            pt2_x = avg_rx

            #print("Shifted Coords: {}, {}".format((pt1_x, pt1_y), (pt2_x, pt2_y)))
            return (pt1_x, pt1_y), (pt2_x, pt2_y)
        else:
            return (pt1_x, pt1_y), (pt2_x, pt2_y)

    def shift_line_x_l(self, point1, point2, avg_lx):
        """ Shifts the left hand line of the internal gap contour in the nevative x direction.
        Outputs a set of shifted coordinates.

        Parameters
        ----------
            point1:
                The first point of the internal gap contour left hand line.
            point2:
                The second point of the internal gap contour left hand line.

        Returns
        -------
            (pt1_x, pt1_y):
                Tuple containing x and y coordinates of shifted point 1.
            (pt2_x, pt2_y):
                Tuple containing x and y coordinates of shifted point 2."""

        pt1_colour = self.get_colour(point1[0], point1[1])
        pt2_colour = self.get_colour(point2[0], point2[1])

        pt1_x, pt2_x = point1[0], point2[0]
        pt1_y, pt2_y = point1[1], point2[1]
        #print("Original Coords: {}, {}".format(point1, point2))
        if pt1_colour == (0, 0, 0) or pt2_colour == (0, 0, 0):
            pt1_x = avg_lx
            pt2_x = avg_lx

            #print("Shifted Coords: {}, {}".format((pt1_x, pt1_y), (pt2_x, pt2_y)))
            return (pt1_x, pt1_y), (pt2_x, pt2_y)
        else:
            return (pt1_x, pt1_y), (pt2_x, pt2_y)

    def deproj_to_pt(self, point1, point2):
        """ Deprojects the pixel data into 3D point data with reference to the camera position.
        Used to find the distances between pixels in millimetres.

        Parameters
        ----------
            point1:
                The first point of the line for which the distance is required to be measured.
            point2:
                The second point of the line for which the distance is required to be measured.

        Returns
        -------
            [dx, dy, dz]:
                A list of the x, y and z distances between the two points in millimetres."""

        pixel1 = [point1[0], point1[1]]
        pixel2 = [point2[0], point2[1]]

        if point1[0] <= 0:
            pixel1[0] = 1
        if point1[0] >= 848:
            pixel1[0] = 847

        if point2[0] <= 0:
            pixel2[0] = 1
        if point2[0] >= 848:
            pixel2[0] = 847

        if point1[1] <= 0:
            pixel1[1] = 1
        if point1[1] >= 480:
            pixel1[1] = 479

        if point2[1] <= 0:
            pixel2[1] = 1
        if point2[1] >= 480:
            pixel2[1] = 479

        point1 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, pixel1, self.aligned_depth_frame.get_distance(pixel1[0], pixel1[1])/self.depth_scale)
        point2 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, pixel2, self.aligned_depth_frame.get_distance(pixel2[0], pixel2[1])/self.depth_scale)

        cv2.circle(self.bg_removed, (pixel1[0], pixel1[1]), 2, (0, 255, 0))
        cv2.circle(self.bg_removed, (pixel2[0], pixel2[1]), 2, (0, 255, 0))

        x1, y1, z1 = point1
        x2, y2, z2 = point2

        dx = math.sqrt(math.pow((x1 - x2), 2))
        dy = math.sqrt(math.pow((y1 - y2), 2))
        dz = math.sqrt(math.pow((z1 - z2), 2))

        return [dx, dy, dz]

    def update(self):
        """ Update function which runs all of the required functionality of the script and draws all required objects to the screen.
        Called every time the screen updates."""

        frames = self.pipeline.wait_for_frames()

        # FRAME ALIGNMENT
        aligned_frames = self.align.process(frames)
        ## Object containing depth frame data of the aligned depth frame (aligned with colour frame).
        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        ## Object containing colour frame data.
        self.colour_frame = aligned_frames.get_color_frame()

        # FILTER THE DEPTH FRAME
        #filtered_depth_frame = self.filter_depth(aligned_depth_frame).
        filtered_depth_frame = self.aligned_depth_frame
        filtered_depth_image = np.asanyarray(filtered_depth_frame.get_data())

        ## Array containing the depth frame data, after being filtered and colourised for viewability.
        self.colourised_filtered_depth = np.asanyarray(self.colouriser.colorize(filtered_depth_frame).get_data())
        ## NumPy array containing colour image data.
        self.colour_image = np.asanyarray(self.colour_frame.get_data())

        # Change this to change how much BG is removed ADAPTIVE
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        clipping_dist_m = self.adapt_depth_clipping(self.aligned_depth_frame)
        self.clipping_dist = clipping_dist_m / self.depth_scale

        # Get Intrinsic depth and colour
        ## Object containing intrinsic depth data used to calculate pixel-to-pixel distances in real-world units.
        self.depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        ## Object containing intrinsic colour data used to calculate pixel-to-pixel distances in real-world units.
        self.colour_intrin = self.colour_frame.profile.as_video_stream_profile().intrinsics
        ## Object containing data with respect to extrinsics of colour data cast to depth data.
        self.depth_colour_extrin = self.aligned_depth_frame.profile.get_extrinsics_to(self.colour_frame.profile)


        # REMOVE BACKGROUND
        grey_colour = 0
        depth_image_3d = np.dstack((filtered_depth_image, filtered_depth_image, filtered_depth_image))
        ## Image object that all objects are drawn on. Background is removed using self.clipping_dist and aligned depth/
        self.bg_removed = np.where((depth_image_3d > self.clipping_dist) | (depth_image_3d <= 0), grey_colour,
                                   self.colour_image)

        # DETECT EDGES
        ## Array object containing thresholded edge frame.
        self.edges = self.canny_edge(self.bg_removed)
        box, rect, internal_contours = self.find_contours(self.edges)
        temp_array = [[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,]]
        #print("CONTOUR INTERNAL: {}".format(internal_contours))
        if internal_contours != temp_array:

            square_gap = max(internal_contours, key=len)

            #print(len(square_gap))

        if type(box) != int:
            mid, w_h, theta = rect
            move_dir_lr = self.get_x_dist(mid, w_h, theta)
            self.check_z_orientation(mid)
            #self.get_move_dir(move_dir_lr)
            #print("Mid: {} Width: {} Height: {} Theta: {}".format(mid, w_h[0], w_h[1], theta))

            # Get ROI points in format:
            #    (1)p_l ---------- p_r(2)
            #       |              |
            #       |              |
            #       |      ROI     |
            #       |              |
            #       |              |
            #       |              |
            #    (3)t_l_ -------- t_r_(4)

            pt3, pt4, pt1, pt2, _tl, _tr, _bl, _br = self.draw_roi(box, theta)

            #Check to see if there is a line of black pixels within the bounding box
            if pt1 and pt4 and pt2 and pt3:
                if theta < -50.:
                    a0, b0, c0, _, _, _ = self.get_line_eq((_tr, _bl), (_tr, _bl))
                else:
                    a0, b0, c0, _, _, _ = self.get_line_eq((_tr, _br), (_tr, _br))
                black_space = []
                for y in range (_tl[1], _bl[1], 2):
                    if a0 != 0:
                        dx = int((_tr[0] - _tl[0]) / 2)
                        x = int((c0 - y)/a0) - dx
                        y = int(y)

                        BGR = self.get_colour(x, y)
                        if BGR == (0, 0, 0):
                            black_space.append((x, y))
        ## Stack of the self.bg_removed image and self.colour_image.
        self.images = np.hstack((self.bg_removed, self.colour_image))

    def get_x_dist(self, mid, wh, theta):
        """ Function to get the direction that the drone is required to move in, only in the x-direction.
        Outputs the direction and distance in pixels to move.

        Parameters
        ----------
            mid:
                Midpoint of the input contour.
            wh:
                Width and height of the input contour.
            theta:
                Angle of rotation through the object y-axis of the object.

        Returns
        -------
            dir:
                Direction of required movement of the drone.
            dx:
                Number of pixels required to be moved."""

        w, h = wh
        if w > 20.:
            mid_x, mid_y = mid

            mid_x = int(mid_x)
            mid_y = int(mid_y)

            screen_mid_x = self.screen_width/2

            dx = screen_mid_x - mid_x

            if dx > 0:
                dir = 'move left'
            elif dx < 0:
                dir = 'move right'
            else:
                dir = 'centered'

            cv2.putText(self.bg_removed, dir, (40, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            return dir, dx

    # Get BGR value of input coordinates
    def get_colour(self, x, y):
        """ Function to get the BGR colour values at specified pixel.
        (0, 0, 0) corresponds to an inactive pixel.

        Parameters
        ----------
            x:
                X-coordinate of pixel.
            y:
                Y-coordinate of pixel.

        Returns
        -------
            (B, G, R):
                Tuple containing (B, G, R) colour data (Blue, Green, Red), between 0 and 255."""

        if y < 0:
            y = 0
        if y > 479:
            y = 479

        if x < 0:
            x = 0
        if x > 847:
            x = 847

        if y < 480 and x < 848:
            B = self.bg_removed[y][x][0]
            G = self.bg_removed[y][x][1]
            R = self.bg_removed[y][x][2]
        else:
            B = G = R = 0

        return (B, G, R)

    # Generate Equation of lines in form a0x + b0y = c0 (1) & a1x + b1y = c1 (2)
    def get_line_eq(self, ln1, ln2):
        """ Function to calculate the equations of two lines.
         In form a0x + b0y = c0 (1) & a1x + b1y = c1 (2).

        Parameters
        ----------
            ln1:
                Line one start and end coordinates.
            ln2:
                Line two start and end coorinates.

        Returns
        -------
            [a0, b0, c0, a1, b1, c1]:
                List of the two lines, to be used in the above format of equations (1) & (2)."""

        # Get X and Y point values from first line
        l1_x0 = ln1[0][0]
        l1_y0 = ln1[0][1]
        l1_x1 = ln1[1][0]
        l1_y1 = ln1[1][1]

        # Get X and Y point values from second line
        l2_x0 = ln2[0][0]
        l2_y0 = ln2[0][1]
        l2_x1 = ln2[1][0]
        l2_y1 = ln2[1][1]

        # Use y = mx + c to find line equation
        if (l1_x1 - l1_x0) != 0:
            m0 = (l1_y1 - l1_y0) / (l1_x1 - l1_x0)
        else:
            m0 = 0

        if (l2_x1 - l2_x0) != 0:
            m1 = (l2_y1 - l2_y0) / (l2_x1 - l2_x0)
        else:
            m1 = 0

        c0 = l1_y0 - (m0 * l1_x0)
        c1 = l2_y0 - (m1 * l2_x0)

        a0 = -1 * m0
        a1 = -1 * m1

        b0 = b1 = 1

        return a0, b0, c0, a1, b1, c1

    # Find intersection of two lines using Cramer's Rule
    def intersection(self, ln1, ln2):
        """ Uses Cramer's Rule to find the intersection point of two lines.
         Outputs the intersection point. Generally used in conjunction with get_line_eq().

        Parameters
        ----------
            ln1:
                Line one input.
            ln2:
                Line two input.

        Returns
        -------
            intercept:
                Intersection point of line one and line two."""

        # Get X and Y point values from first line
        l1_x0 = ln1[0][0]
        l1_y0 = ln1[0][1]
        l1_x1 = ln1[1][0]
        l1_y1 = ln1[1][1]

        # Get X and Y point values from second line
        l2_x0 = ln2[0][0]
        l2_y0 = ln2[0][1]
        l2_x1 = ln2[1][0]
        l2_y1 = ln2[1][1]

        # Use y = mx + c to find line equation
        if (l1_x1 - l1_x0) != 0:
            m0 = (l1_y1 - l1_y0) / (l1_x1 - l1_x0)
        else:
            m0 = 0

        if (l2_x1 - l2_x0) != 0:
            m1 = (l2_y1 - l2_y0) / (l2_x1 - l2_x0)
        else:
            m1 = 0

        c0 = l1_y0 - (m0 * l1_x0)
        c1 = l2_y0 - (m1 * l2_x0)

        a0 = -1 * m0
        a1 = -1 * m1

        b0 = b1 = 1

        # If a0x + b0y = c0 (1) & a1x + b1y = c1 (2),
        # Then (1)*b1: a0b1x + b0b1y = c0b1 (3) & (2)*b0: a1b0x + b1b0y = c1b0 (4)
        # (3) - (4): (a0b1 - a1b0)x + (b0b1 - b1b0)y = c0b1 - c1b0
        # (3) - (4): (a0b1 - a1b0)x = c0b1 - c1b0
        # This gives intersection x coord.
        # Sub x coord back into eq (3) to find y

        if ((a0 * b1) - (a1 * b0)) != 0:
            x = ((c0 * b1) - (c1 * b0)) / ((a0 * b1) - (a1 * b0))
            y = (c0 * b1) - (a0 * b1 * x)

            intercept = (x, y)
        else:
            intercept = ('NA', 'NA')

        return intercept

    # Show the OpenCV Windows
    def show_window(self):
        """ Function used to show the 'Realsense' window and the 'Edge' window
        'Realsense' - Shows processed image with all drawn contours and eliminated background, next to the colourised
                      depth stream.
        'Edge' - Shows thresholded edge detection image."""

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', self.images)
        cv2.imshow('Edge', self.edges)

        cv2.waitKey(1)