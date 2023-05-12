""" A simple robotics navigation code including SLAM, exploration, planning"""

import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(
            self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros(
            (int(self.x_max_map), int(self.y_max_map)))

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

        self.path = False
        


    def _conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """
        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def _conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """
        x_world = self.x_min_world + x_map *  self.resolution
        y_world = self.y_min_world + y_map *  self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world

    def add_map_line(self, x_0, y_0, x_1, y_1, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """
        # convert to pixels
        x_start, y_start = self._conv_world_to_map(x_0, y_0)
        x_end, y_end = self._conv_world_to_map(x_1, y_1)
        if x_start < 0 or x_start >= self.x_max_map or y_start < 0 or y_start >= self.y_max_map:
            return
        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return
        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        # add value to the points
        self.occupancy_map[points[0], points[1]] += val  

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """
        x_px, y_px = self._conv_world_to_map(points_x, points_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.x_max_map),
                                np.logical_and(y_px >= 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val


    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        laser_dist = lidar.get_sensor_values()
        laser_angle = lidar.get_ray_angles()
        reserve = laser_dist < lidar.max_range #laser_dist.max()
        laser_dist = laser_dist[reserve]
        laser_angle = laser_angle[reserve]
        # Estimer les positions des d´etections du laser dans le repère global
        laser_x_world = pose[0] + laser_dist*np.cos(pose[2]+laser_angle)
        laser_y_world = pose[1] + laser_dist*np.sin(pose[2]+laser_angle)
        # Convertir ces positions dans le repère de la carte
        laser_x_map, laser_y_map = self._conv_world_to_map(laser_x_world, laser_y_world)
        # supprimer les points hors de la carte
        laser_x_map = [x for x in laser_x_map if x<= self.occupancy_map.shape[0] and x>=0]
        laser_y_map = [y for y in laser_y_map if y<= self.occupancy_map.shape[1] and y>=0]
        # Lire et additionner les valeurs
        score = np.sum(np.exp(self.occupancy_map[laser_x_map, laser_y_map]))
        
        return score

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
            # changement de repère par une transformation matricielle
            # corrected_pose[:2] = odom[:2]*[[np.cos(theta), np.sin(theta)],[ -np.sin(theta),  np.cos(theta)]] + self.odom_pose_ref[:2]
            # corrected_pose[2] = odom[2] + self.odom_pose_ref[2]
            # theta = odom_pose_ref[2]
        corrected_pose = odom + odom_pose_ref
 
        return corrected_pose

    def localise(self, lidar, odom):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4
        # Calculez le score du scan laser avec la position de r´ef´erence actuelle de l’odom´etrie
        best_score = self.score(lidar, self.get_corrected_pose(odom))
        odom_pose_ref = self.odom_pose_ref
        sigma = [0.7, 0.6, 0.02]                            
        # Tirez 10 fois
        for i in range(100): 
            # Tirez un offset aléatoire
            offset = np.random.normal(odom_pose_ref, sigma)
            # ajoutez le à la position de référence de l’odométrie
            pose_world = self.get_corrected_pose(odom, offset)
            # Calculez le score du scan laser avec cette nouvelle position de r´ef´erence de l’odom´etrie
            score_tmp = self.score(lidar, pose_world)
            # Si le score est meilleur, m´emorisez ce score et la nouvelle position de r´ef´erence
            if score_tmp > best_score:
                best_score = score_tmp
                odom_pose_ref = offset
        #print("vecteur de correction: ", odom_pose_ref)
        #print("score: ", best_score)
        if best_score > 20000:
            self.odom_pose_ref = odom_pose_ref

   
    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
         
        # convert polar coordinates of laser scan to absolute position
        laser_dist = lidar.get_sensor_values()
        laser_angle = lidar.get_ray_angles() 
        laser_coor_x = pose[0] + laser_dist*np.cos(pose[2] + laser_angle)
        laser_coor_y = pose[1] + laser_dist*np.sin(pose[2] + laser_angle)
        
        # mise à jour des points sur la ligne entre robot et points détectés avec une probabilité faible
        # mise à jour des points détectés avec une probabilité forte
        # mettre une zone pour verrouiller les erreurs
        for i in range(360):
            self.add_map_line(pose[0], pose[1], laser_coor_x[i], laser_coor_y[i], -2)
            self.add_map_line(pose[0]+(laser_coor_x[i]-pose[0])*0.95, pose[1]+(laser_coor_y[i]-pose[1])*0.95, laser_coor_x[i], laser_coor_y[i], 2)
        self.add_map_points(laser_coor_x, laser_coor_y,4)

        # seuillage
        seuil = 6
        self.occupancy_map[self.occupancy_map > seuil] = seuil
        self.occupancy_map[self.occupancy_map < -seuil] = -seuil

    def get_neighbors(self, current):
        """
        Trouver les voisins 
        current: coordonnées x y de la position du robot
        """
        x = current[0]
        y = current[1]
        neighbors = [[x-1, y+1], [x, y+1], [x+1, y+1], 
                     [x-1, y], [x+1, y], 
                     [x-1, y-1], [x, y-1], [x+1, y-1]]
        return neighbors
    
    def reconstruct_path(self,cameFrom, current):
        """build the road from start to goal when A star has finished the plan"""
        total_path = [current]
        index = current[0]*self.occupancy_map.shape[0] + current[1]
        while index in cameFrom.keys():
            current = cameFrom[index]
            total_path.insert(0,current)
            index = current[0]*self.occupancy_map.shape[0] + current[1]
        return total_path

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """
        # TODO for TP5

        # initialization
        openSet = [start]
        closeSet = []
        cameFrom = {}
        g = np.full_like(self.occupancy_map, np.inf)
        g[start] = 0
        f = np.full_like(self.occupancy_map, np.inf)
        f[start] = np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)

        while openSet:
            openSet_arr = np.array(openSet)
            f_tmp = f[openSet_arr[:,0], openSet_arr[:,1]]
            ind_cur = np.argmin(f_tmp)
            current = openSet[ind_cur]
            self.occupancy_map[current[0], current[1]] = 1.8
            self.display2(self.odom_pose_ref)
            # if found the path return
            if current[0] == goal[0] and current[1] == goal[1]:
                total_path = self.reconstruct_path(cameFrom, current)
                self.path = np.array(total_path)
                return total_path
            openSet.remove(current)
            closeSet.append(current)

            neighbours = self.get_neighbors(current)
            for neighbour in neighbours:
                if neighbour in closeSet or self.occupancy_map[neighbour[0], neighbour[1]] > 1:
                    continue
                tentative_g = g[current[0], current[1]] + np.sqrt((neighbour[0]-current[0])**2 + (neighbour[1]-current[1])**2)
                if tentative_g < g[neighbour[0], neighbour[1]]:
                    cameFrom[neighbour[0]*self.occupancy_map.shape[0] + neighbour[1]] = current
                    g[neighbour[0], neighbour[1]] = tentative_g
                    f[neighbour[0], neighbour[1]] = tentative_g + np.sqrt((goal[0]-neighbour[0])**2 + (goal[1]-neighbour[1])**2)
                    if neighbour not in openSet:
                        openSet.append(neighbour)
        return False



    def display(self, robot_pose):
        """
        Screen display of map and robot pose, using matplotlib
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world, self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(robot_pose[0], robot_pose[1], delta_x, delta_y,
                  color='red', head_width=5, head_length=10, )

        # plt.show()
        plt.pause(0.001)

    def display2(self, robot_pose):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

    def save(self, filename):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world,
                           self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + '.png')

        with open(filename + ".p", "wb") as fid:
            pickle.dump({'occupancy_map': self.occupancy_map,
                         'resolution': self.resolution,
                         'x_min_world': self.x_min_world,
                         'x_max_world': self.x_max_world,
                         'y_min_world': self.y_min_world,
                         'y_max_world': self.y_max_world}, fid)

    def load(self, filename):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        # TODO


