"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import reactive_obst_avoid
from control import potential_field_control


# Definition of our robot controller
class MyRobotSlam(RobotAbstract): #类的继承
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        self._size_area = (800, 800)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0,0,0])

    def control(self):
        """
        Main control function executed at each time step
        """
        start_plan = 400
        self.counter += 1
        
        if self.counter == 1:
            print("Start moving by avoiding obstacles ...")
        self.tiny_slam.update_map(self.lidar(), self.odometer_values())
        # afficher la carte
        self.tiny_slam.display(self.odometer_values())
        #filename = "cartographie_test"
        #self.tiny_slam.save(filename)
        
        if self.counter%20 == 0 and self.counter!=0:
        # mettre à jour la position de r´ef´erence de l’odom´etrie si le score depasse le seuil
            TinySlam.localise(self.tiny_slam,self.lidar(),TinySlam.get_corrected_pose(self.tiny_slam,self.odometer_values()))

        # first, go random steps to perform obstacle avoidance
        if self.counter < start_plan:
            command = reactive_obst_avoid(self.lidar())

        # at a certain point, stop and plan its way home (point de départ)
        elif self.counter == start_plan:
            print("Start planning ...")
            start = self.tiny_slam.get_corrected_pose(self.odometer_values())
            start_map = self.tiny_slam._conv_world_to_map(start[0], start[1])
            goal_map = self.tiny_slam._conv_world_to_map(0, 0)
            self.tiny_slam.plan(start_map, goal_map)
            print("trajectoire planifiée :", self.tiny_slam.path)
            for point in self.tiny_slam.path:
                pose = [point[0]- self._size_area[0]/2, point[1]-self._size_area[1]/2, 0]
                self.tiny_slam.display2(pose)
            command = {"forward": 0,
               "rotation": 0}
            

            print("Start following the planned path ...")

        # use potential field to follow the trajectory planned by A*
        else:
            #goal = self.tiny_slam._conv_map_to_world(trajectory[0, 0], trajectory[0, 1])
            #command = potential_field_control(self.lidar(), self.odometer_values(), [goal[0], goal[1], 1])
             # l'origine du repère est au centre de la carte

            if self.tiny_slam.path is False:
                command = reactive_obst_avoid(self.lidar())
            else:
                goal = self.tiny_slam._conv_map_to_world(self.tiny_slam.path[0, 0], self.tiny_slam.path[0, 1])
                command = potential_field_control(self.lidar(), self.odometer_values(), [goal[0], goal[1], 1])

                if self.counter % 10 == 0:
                    if self.tiny_slam.path.shape[0] == 1:
                        print("The robot has arrived.")
                        command = {"forward": 0,
                            "rotation": 0}
                    else:
                        self.tiny_slam.path = self.tiny_slam.path[1:]
        return command
