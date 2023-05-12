""" A set of robotics control functions """
import random
import numpy as np
import math

# initiate a random reaction
rotation_dir = -1 # random.uniform(-1, 1)

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    global rotation_dir
    laser_dist = lidar.get_sensor_values()
    speed = np.zeros(2)
    if min(laser_dist[90:240]) < 30 :
        speed[0] = 0.0                  # vitesse linéaire
        speed[1] = rotation_dir*0.2                # vitesse de rotation
    else :
        speed[0] = 0.2
        speed[1] = 0.0

    command = {"forward": speed[0],
               "rotation": speed[1]}
    
    return command



def potential_field_control(lidar, pose, goal):
    K_goal = 1 # champ potentiel attractif
    K_obs = -100 # champ potentiel répulsif
    d_safe = 70   # non nul
    laser_dist = lidar.get_sensor_values()
    laser_angle = lidar.get_ray_angles() 
    speed = np.zeros(2)
    
        
    # champ attractif
    dist = np.linalg.norm(goal[:2] - pose[:2])
    f_attire = K_goal*(goal[:2]-pose[:2])/dist   
    # champ répulsif
    index_min = np.argmin(laser_dist)
    d_obs =  lidar.get_sensor_values()[index_min]    # d(q - q_obs)
    angle_obs =  lidar.get_ray_angles()[index_min]   # par rapport à l'avant du robot
    
    # champ résultant
    if (d_obs>d_safe):
        f_repu = [0, 0]
    else :
        f_repu = K_obs*(1/d_obs - 1/d_safe)*d_obs*np.array([np.cos(angle_obs+pose[2]), np.sin(angle_obs+pose[2])])/(d_obs**3)
    force = f_attire + f_repu
  

    # calculer la vitesse angulaire 
    _pose = [np.cos(pose[2]), np.sin(pose[2])]  # on crée un vecteur de pose
    the_norm = np.linalg.norm(force)
    # produit vectoriel
    rho = np.rad2deg(np.arcsin(np.cross(_pose, force) / the_norm))  
    # produit scalaire
    alpha = np.arccos(np.dot(_pose, force)/the_norm)  # l'angle que le robot doit tourner   # arccos rend une valeur entre 0 et pi                                                 
    if rho < 0 :
        alpha = -1*alpha                  # - clock-wise  ,    + anti clock wise
        
    linear = np.linalg.norm(force)
    speed[0] = 0.2 if linear > 1 else linear*0.2
    speed[1] = 0.5*alpha/np.pi 
  
    command = {"forward": speed[0],
               "rotation": speed[1]}
    return command
  

    