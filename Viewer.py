#Class for the realsense image processing

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
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        self.screen_width = 848
        self.screen_height = 480
        config.enable_stream(rs.stream.depth, self.screen_width, self.screen_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.screen_width, self.screen_height, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.colouriser = rs.colorizer()
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        #Points array for finding dead space
        self.dead_points = []
        #Contours array for setting contour
        self.active_contours = []

        # Change this to change how much BG is removed initially
        clipping_dist_m = 5
        self.clipping_dist = clipping_dist_m / depth_scale

        ### INITIALISE FLIGHT CONTROLL VARIABLES ###

        self.dead_space = False
        self.pillar_detected = False

        self.verified_deployment_area = False

        self.direction_to_move = ''
        self.distance_to_move = 0
        self.distance_from_pillar = 0.
        self.deployment_area_coordinates = []

        self.internal_contour = False
        self.external_contour = False

        self.contour_shrink_sf = 1.

        self.deploy_tl = (0, 0)
        self.deploy_tr = (0, 0)
        self.deploy_bl = (0, 0)
        self.deploy_br = (0, 0)

    def adapt_depth_clipping(self, depth_image):
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
        #print("Adapt Depth Clipping: {}".format(time1))

        return closest_obj_tol

    def get_move_dir(self, lr_dir):
        drone_dir = ' '
        if lr_dir == 'move left':
            drone_dir = 'L'
        elif lr_dir == 'move right':
            drone_dir = 'R'
        elif lr_dir == 'centered':
            drone_dir = 'X'

    def get_deploy_coordinates(self):
        pass

    def do_nothing(self):
        pass

    # Filter the depth image to remove a bit of noise/fill holes etc [CURRENTLY REDUNDANT]
    def filter_depth(self, d_frame):
        #e1 = cv2.getTickCount()
        frame = d_frame

        #self.decimation.set_option(rs.option.filter_magnitude, 1)

        #self.spatial.set_option(rs.option.filter_magnitude, 5)
        #self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        #self.spatial.set_option(rs.option.filter_smooth_delta, 50)

        #frame = self.decimation.process(frame)
        #frame = self.depth_to_disparity.process(frame)
        #frame = self.spatial.process(frame)
        #frame = self.disparity_to_depth.process(frame)
        #frame = self.hole_filling.process(frame)
        #frame = frame.as_depth_frame()
        #e2 = cv2.getTickCount()
        #time1 = (e2 - e1) / cv2.getTickFrequency()
        #print("Filter Depth: {}".format(time1))
        return frame

    # Detect edges in the image [CHECK THRESHOLDS]
    def canny_edge(self, image):

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

    def find_contours(self, edges):
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
                        self.internal_contour = True
                    else:
                        square_gap = [0, 0, 0, 0]
                        contours_int = [[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,]]
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
                                self.internal_contour = True
                            else:
                                square_gap = [0, 0, 0, 0]
                                contours_int = [[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,]]
                            #print("Int Cnt Else: {}".format(int_cnt))
                            del (self.active_contours[0])
                            break

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

                cv2.drawContours(self.bg_removed, [box], 0, (0,255,0), 2)

                if self.internal_contour:

                    int_cnt_perim = cv2.arcLength(square_gap, True)

                    if int_cnt_perim > 40.:

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

                        cv2.circle(self.bg_removed, (c_x, c_y), 7, (255, 255, 255), -1)

                        gap_rect = cv2.minAreaRect(square_gap)

                        gap_box = cv2.boxPoints(gap_rect)
                        gap_box = np.int0(gap_box)

                        cv2.drawContours(self.bg_removed, [gap_box], 0, (255, 0, 255), 2)
                        #cv2.drawContours(self.bg_removed, [cnt_scaled], 0, (0, 0, 255), 2)
                        if self.verified_deployment_area:

                            cv2.drawContours(self.bg_removed, [scaled_box], 0, (0, 255, 255), 2)
                            self.get_deployment_coords(self.deploy_tl, self.deploy_tr, self.deploy_bl, self.deploy_br)

                        #print("Gap Box Points: {}".format(gap_box))

                        #self.shrink_gap_boundaries(gap_box)

                '''
                rows, cols = im.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                cv2.line(self.bg_removed, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
                '''

                return box, rect, contours_int

        false_return = [[0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,], [0, 0, 0, 0,]]

        return 0, 0, false_return

    def get_deployment_coords(self, tl, tr, bl, br):

        self.deploy_tl = tl
        self.deploy_tr = tr
        self.deploy_bl = bl
        self.deploy_br = br

        self.deployment_area_coordinates = [self.deploy_tl, self.deploy_tr, self.deploy_bl, self.deploy_br]

        print("Deployment Coords: {}".format(self.deployment_area_coordinates))

        return self.deployment_area_coordinates

    def scale_internal_contour(self, centre, dims):
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

        print("Status: {} Coords: ({}), ({}), ({}), ({})".format(self.verified_deployment_area, tl, tr, bl, br))

        if self.verified_deployment_area == False:
            self.contour_shrink_sf -= 0.05

        return tl, tr, bl, br

        #cv2.circle(self.bg_removed, tl, 3, (255, 255, 255), -1)
        #cv2.circle(self.bg_removed, tr, 3, (255, 255, 255), -1)
        #cv2.circle(self.bg_removed, bl, 3, (255, 255, 255), -1)
        #cv2.circle(self.bg_removed, br, 3, (255, 255, 255), -1)

    '''
    def shrink_gap_boundaries(self, gap_box):
        point_list = []
        shrunk_points = []

        pt1, pt2, pt3, pt4 = gap_box

        point_list.append(pt1)
        point_list.append(pt2)
        point_list.append(pt3)
        point_list.append(pt4)

        #Sorted into order :TL, BL, TR, BR
        sorted_point_list = sorted(point_list, key=operator.itemgetter(0, 1))

        #Begin by shrinking square by 10%


        #print("Original: {}".format(point_list))
        #print("Sorted: {}".format(sorted_point_list))
        self.times_shrunk = 0
        shrunk_points = self.shrink_5(sorted_point_list)

        while self.verified_deployment_area == False:
            self.verify_deployment_area(shrunk_points)

            shrunk_points = self.shrink_5(shrunk_points)

        self.deployment_area_coordinates = shrunk_points

        #print("Deployment Area Coordinates: {}".format(self.deployment_area_coordinates))

        shrunk_points = np.int0(shrunk_points)

        for point in shrunk_points:
            cv2.circle(self.bg_removed, (point[0], point[1]), 3, (0, 0, 255))
    '''
    def verify_deployment_area(self, shrunk_points):
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

    '''
    def shrink_5(self, sorted_points):

        temp_points = []
        temp_element = []
        tuple_no = 0
        element_no = 0

        for tuple in sorted_points:
            element_no = 0
            temp_element = []
            for element in tuple:
                if tuple_no == 0:
                    if element_no == 0:
                        element += 10
                    elif element_no == 1:
                        element += 10
                    else:
                        break

                elif tuple_no == 1:
                    if element_no == 0:
                        element += 10
                    elif element_no == 1:
                        element -= 10
                    else:
                        break

                elif tuple_no == 2:
                    if element_no == 0:
                        element -= 10
                    elif element_no == 1:
                        element += 10
                    else:
                        break

                elif tuple_no == 3:
                    if element_no == 0:
                        element -= 10
                    elif element_no == 1:
                        element -= 10
                    else:
                        break

                else:
                    break

                temp_element.insert(element_no, element)
                element_no += 1

            temp_points.insert(tuple_no, temp_element)
            tuple_no += 1

        self.times_shrunk += 1
        print(self.times_shrunk)
        #print("Temp Points: {}".format(temp_points))

        return temp_points
    '''
    def separate_contours(self, contours, hierarchy):
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
        if ext_cnt:
            self.external_contour = True

        #print("INT: {} EXT: {}".format(int_cnt, ext_cnt))
        return int_cnt, ext_cnt

    def draw_roi(self, box, theta):
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

    def update(self):

        frames = self.pipeline.wait_for_frames()

        # FRAME ALIGNMENT
        aligned_frames = self.align.process(frames)

        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        self.colour_frame = aligned_frames.get_color_frame()

        # FILTER THE DEPTH FRAME
        #filtered_depth_frame = self.filter_depth(aligned_depth_frame)
        filtered_depth_frame = self.aligned_depth_frame
        filtered_depth_image = np.asanyarray(filtered_depth_frame.get_data())

        self.colourised_filtered_depth = np.asanyarray(self.colouriser.colorize(filtered_depth_frame).get_data())

        self.colour_image = np.asanyarray(self.colour_frame.get_data())

        # Change this to change how much BG is removed ADAPTIVE
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        clipping_dist_m = self.adapt_depth_clipping(self.aligned_depth_frame)
        self.clipping_dist = clipping_dist_m / depth_scale

        # Get Intrinsic depth
        self.depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        # REMOVE BACKGROUND
        grey_colour = 0
        depth_image_3d = np.dstack((filtered_depth_image, filtered_depth_image, filtered_depth_image))
        self.bg_removed = np.where((depth_image_3d > self.clipping_dist) | (depth_image_3d <= 0), grey_colour,
                                   self.colour_image)

        # DETECT EDGES
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
            self.get_move_dir(move_dir_lr)
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

            #print("        #    {} ---------- {}".format(pt1, pt2))
            #print("        #     |                        |")
            #print("        #     |                        |")
           # print("        #     |    ROI                 |")
            #print("        #     |                        |")
            #print("        #     |                        |")
           # print("        #    {} ---------- {}".format(pt3, pt4))

            #Check to see if there is a line of black pixels within the bounding box
            if pt1 and pt4 and pt2 and pt3:
                if theta < -50.:
                    a0, b0, c0, _, _, _ = self.get_line_eq((_tr, _bl), (_tr, _bl))
                else:
                    a0, b0, c0, _, _, _ = self.get_line_eq((_tr, _br), (_tr, _br))
               # print("Pt1: {} Pt2: {}\nEq: {}x + {}y = {}".format(_tl, _bl, a0, b0, c0))
               # print("TL: {} TR: {} BL: {} BR: {}, THETA: {}".format(_tl, _tr, _bl, _br, theta))
                black_space = []
                for y in range (_tl[1], _bl[1], 2):
                    if a0 != 0:
                        dx = int((_tr[0] - _tl[0]) / 2)
                        x = int((c0 - y)/a0) - dx
                        y = int(y)

                        BGR = self.get_colour(x, y)
                        if BGR == (0, 0, 0):
                            black_space.append((x, y))

                        #cv2.circle(self.bg_removed, (x, y), 4, (0, 255, 0), 2)
               # print(black_space)
                #for coord in black_space:
                    #cv2.circle(self.bg_removed, coord, 4, (255, 255, 0), 2)
            #self.find_dead_space_flat(pt1, pt4)

        self.images = np.hstack((self.bg_removed, self.colour_image))

    def get_x_dist(self, mid, wh, theta):
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

            return dir

    def find_dead_space_with_bearings(self):
        pass

    def find_dead_space_flat(self, pt1, pt4):

        # The following 8 lines of code ensure that the processed points will never be (0, 0).
        # This works by:
        # 1 - Initialise an array with 11 points before the loop starts.
        # 2 - Work backwards through the array.
        # 3 - If the most recent point in the array does not equal (0, 0) use these points.
        # 4 - If the most recent point in the array equals (0, 0), work back until a non-zero point is found, and use that.

        self.dead_points.append((pt1, pt4))
        if len(self.dead_points) > 10:
            for i in range(0, 10):
                if self.dead_points[10 - i] != (0, 0):
                    pt1 = self.dead_points[10 - i][0]
                    pt4 = self.dead_points[10 - i][1]
                    del (self.dead_points[0])
                    break

            # Redundancy check just to make sure that a (0, 0) point has not made it through the previous safety check.
            if pt1 and pt4:

                # Begin by defining the start and end x and y of the ROI as a square, contracted by 20 pixels.
                roi_startx = pt1[0] + 20
                roi_starty = pt1[1] + 20

                roi_endx = pt4[0] - 20
                roi_endy = pt4[1] - 20

                # Initialise arrays to count the amount of pixels scanned, the amount of black pixels found, and an
                # array to store the colours found.
                pixels_scanned = 0
                pixels_black = 0
                colour_array = []

                # Step through the defined ROI in increments of 4 to save processing time.
                for _x in range(roi_startx, roi_endx,4):
                    for _y in range(roi_starty, roi_endy,4):
                        # Check the colour of the defined pixel, and append this into the colours array.
                        BGR = self.get_colour(_x, _y)
                        colour_array.append(BGR)

                        # If the analysed pixel is black, increment the counter.
                        if BGR == (0, 0, 0):
                            pixels_black += 1

                        # Increment the counter for the amount of pixels scanned every iteration of the loop.
                        pixels_scanned += 1

                # If every pixel that was scanned in the previous loop was a black pixel, the dead space has been found.
                # If this is the case, print 'Dead Space' on the screen, and toggle the dead space boolean. Otherwise,
                # make sure that the dead space boolean remains low, so as to not trigger the initial stages of deployment.
                if pixels_scanned == pixels_black:
                    cv2.putText(self.bg_removed, 'Dead Space', (roi_startx - 30, roi_starty - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.dead_space = True
                else:
                    self.dead_space = False
               # print("Dead Space Status: {}".format(self.dead_space))
            #print("Top Left: {} \n Size: {} \n Theta: {}".format(top_left, size, theta))

    def dead_space_detected(self):
        return self.dead_space

    def get_lr_boundaries(self):

        lines = []

        if self.line_l:
            lines.append(self.line_l)
        if self.line_r:
            lines.append(self.line_r)

        return lines

    # Define a ROI above input line
    def setROI(self, line):

        x0 = int(line[0][0])
        y0 = int(line[0][1])

        x1 = int(line[1][0])
        y1 = int(line[1][1])

        cv2.rectangle(self.colour_image, (x0, y0 - 20), (x1, y1 - 100), (0, 255, 255), 2)
        cv2.rectangle(self.bg_removed, (x1, y0 - 20), (x0, y1 - 100), (0, 255, 255), 2)

        _y0 = y0 - 20
        _y1 = y1 - 100

        # Sort so that first tuple contains smallest numbers for for loop to run properly
        if x0 < x1 and _y0 < _y1:
            roi = ((x0, _y0), (x1, _y1))
        elif x0 < x1 and _y1 < _y0:
            roi = ((x0, _y1), (x1, _y0))
        elif x1 < x0 and _y0 < _y1:
            roi = ((x1, _y0), (x0, _y1))
        else:
            roi = ((x1, _y1), (x0, _y0))

        return roi

    # Get BGR value of input coordinates
    def get_colour(self, x, y):

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

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', self.images)
        cv2.imshow('Edge', self.edges)

        cv2.waitKey(1)