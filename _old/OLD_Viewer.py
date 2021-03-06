'''
        # DRAW LINES WHERE DETECTED
        # Possibly store all lines in an array, and check if any lines are intersecting.
        # If a horizontal line is intersecting two vertical lines that are however far
        # apart the two sides of the column are, compare this with the depth in the gap
        # above to confirm if the top of the support has been reached.
        # IF TOP LINE IS NOT BEING FOUND, ALTER HOUGHLINES THRESHOLD

        lines = cv2.HoughLines(self.edges, 1, np.pi / 180, 100, None, 0, 0)

        points = []
        line_rhs_points = []
        line_rhs_angle = []
        line_lhs_points = []
        line_lhs_angle = []
        line_horiz_points = []
        line_horiz_angle = []

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)

                x0 = a * rho
                y0 = b * rho

                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

                # Find all lines on the right hand side of square (angle > 175 deg)
                # >175
                if math.degrees(theta) > 150.:
                    point = (pt1, pt2)
                    line_rhs_points.append(point)
                    line_rhs_angle.append(math.degrees(theta))

                # Find all lines on the left hand side of square (angle < 2 deg)
                # < 2.
                if 0.01 < math.degrees(theta) < 30.:
                    point = (pt1, pt2)
                    line_lhs_points.append(point)
                    line_lhs_angle.append(math.degrees(theta))

                # Find all lines on the horizontal of square (88 < angle < 92 deg)
                # Determine a differential value of the angle by finding the modulus of the angle - 90.0
                # 88. 92.
                if 70. < math.degrees(theta) < 110.:
                    angle = math.sqrt(math.pow((math.degrees(theta) - 90.), 2))
                    point = (pt1, pt2)
                    line_horiz_points.append(point)
                    line_horiz_angle.append(angle)

        orientations = []
        # Find most vertical left hand line, draw and append to points array
        if len(line_lhs_angle) != 0:
            self.min_left_angle = np.amin(line_lhs_angle)
            min_left_angle_index = line_lhs_angle.index(self.min_left_angle)

            #cv2.line(self.colour_image, line_lhs_points[min_left_angle_index][0],
                     #line_lhs_points[min_left_angle_index][1], (0, 255, 0), 2)

            points.append(line_lhs_points[min_left_angle_index])
            orientations.append('l')

        # Find most vertical right hand line, draw and append to points array
        if len(line_rhs_angle) != 0:
            self.max_right_angle = np.amax(line_rhs_angle)
            max_right_angle_index = line_rhs_angle.index(self.max_right_angle)

            #cv2.line(self.colour_image, line_rhs_points[max_right_angle_index][0],
                     #line_rhs_points[max_right_angle_index][1], (0, 0, 255), 2)

            points.append(line_rhs_points[max_right_angle_index])
            orientations.append('r')

        # Find most horizontal line, draw and append to points array
        if len(line_horiz_angle) != 0:

            left_angle = self.min_left_angle
            right_angle = 180. - self.max_right_angle

            avg_vert_angle = (left_angle + right_angle)/2

            req_horiz_angle = avg_vert_angle
            found_horiz_angle = min(line_horiz_angle, key=lambda x:abs(x-req_horiz_angle))
            max_horiz_angle_index = line_horiz_angle.index(found_horiz_angle)

            #print("Required H Angle: {}".format(req_horiz_angle))
            #print("Horizontal Line Angle: {}".format(found_horiz_angle))
            #print("Left Line Angle: {}".format(left_angle))
            #print("Right Line Angle: {}".format(right_angle + 180.))

           # cv2.line(self.colour_image, line_horiz_points[max_horiz_angle_index][0],
                     #line_horiz_points[max_horiz_angle_index][1],
                    # (0, 0, 255), 2)

            points.append(line_horiz_points[max_horiz_angle_index])
            orientations.append('h')

        # Call intercept function and append into array, ignoring all invalid points
        inter_points = []

        if len(points) == 3:
            l_index = orientations.index('l')
            r_index = orientations.index('r')
            h_index = orientations.index('h')

            self.line_l = points[l_index]
            self.line_r = points[r_index]
            self.line_h = points[h_index]

            inter_lh = self.intersection(self.line_l, self.line_h)
            if inter_lh != ('NA', 'NA'):
                inter_points.append(inter_lh)

            inter_rh = self.intersection(self.line_r, self.line_h)
            if inter_rh != ('NA', 'NA'):
                inter_points.append(inter_rh)

            top_lines = self.intercept_and_topline(inter_points)

            # Get ROI and process data within it
            if len(top_lines) != 0:
                _roi = self.setROI(top_lines[0])

                # Loop through ROI and record colour of each pixel
                roi_colour = 0
                size_iterator = 0

                if int(((_roi[1][0] - 2) - (_roi[0][0] + 2))/10) > 0:
                    x_step = int(((_roi[1][0] - 2) - (_roi[0][0] + 2))/10)
                else:
                    x_step = 1
                if int(((_roi[1][1] - 2) - (_roi[0][1] + 2)) / 10) > 0:
                    y_step = int(((_roi[1][1] - 2) - (_roi[0][1] + 2))/10)
                else:
                    y_step = 1

                for x in range(_roi[0][0] + 2, _roi[1][0] - 2, x_step):
                    for y in range(_roi[0][1] + 2, _roi[1][1] - 2, y_step):
                        if self.get_colour(x, y) == (153, 153, 153):
                            roi_colour += 1
                        size_iterator += 1

                if roi_colour == size_iterator:
                    cv2.rectangle(self.bg_removed, (_roi[0][0] - 30, _roi[0][1] - 30),
                                  (_roi[1][0] + 30, _roi[1][1] + 30), (0, 0, 255), 4)
                    cv2.putText(self.bg_removed, 'Dead Space', (_roi[0][0] - 30, _roi[0][1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            '''





'''
        # Only proceed if there are 10 contours stored
        if len(self.active_contours) == storage_length:
            # Append perims and areas into respective arrays
            for i in range(0, storage_length):
                perims.append(cv2.arcLength(self.active_contours[i][0], True))
                areas.append(cv2.contourArea(self.active_contours[i][0]))

            # Find the current area and average of last 10 areas
            avg_area = mean(areas)
            curr_area = cv2.contourArea(self.active_contours[i][0])


            if len(areas) >= storage_length:
                # If the current area is greater and average, or within 15% less than average, use the most recently
                # stored contour
                if (curr_area > avg_area) or ((avg_area - (avg_area*0.15)) < curr_area):
                    contours = self.active_contours[-1]

                    #Remove the oldest contour from the list
                    del(self.active_contours[0])

                else:
                    # If the contour area is outside the set threshold, scan the last 10 contours to find a suitable one
                    for j in range(0, storage_length):
                        if areas[storage_length - 1 - j] > avg_area or ((avg_area - (avg_area*0.15)) < areas[storage_length - 1 - j]):
                            # If a contour area is within the threshold, use this contour.
                            contours = self.active_contours[storage_length - 1 - j]
                            del (self.active_contours[0])
                            break
        '''

'''
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
            for _x in range(roi_startx, roi_endx, 4):
                for _y in range(roi_starty, roi_endy, 4):
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
                # print("Top Left: {} \n Size: {} \n Theta: {}".format(top_left, size, theta))
'''