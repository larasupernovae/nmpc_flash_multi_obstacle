#!/usr/bin/env python3

"""
NMPC Controller for DJI Matrice 100 (UAV Control)
-----------------------------------------
This module contains the MPCController class for Non-Linear Model Predictive Control (NMPC)
of a quadrotor. It integrates trajectory generation, dynamic modeling, and solving optimization 
problems using Nonlinear MPC (NMPC) with and without static and dynamic obstacle avoidance.

Author: Tommy Persson and Lara Laban
Created on: 24 March 2024

License: MIT License
Copyright (c) 2024 Tommy Persson and Lara Laban

This project is developed under the Lund University / Linköping University / WASP (WARA PS) initiative.

For full license details, see the LICENSE file in the root directory of this project.
"""

# ========================================================================================
# Import Libraries
#
# This section includes all the necessary imports for ROS, math operations, data handling,
# geometry messages, visualization messages, and other necessary utilities.
# ========================================================================================

import rospy
import math
import sys

from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import pandas as pd

from geometry_msgs.msg import PointStamped, QuaternionStamped, Vector3Stamped, Vector3, Point, Quaternion
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64, Header, ColorRGBA, Empty
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Joy, LaserScan, BatteryState

from optparse import OptionParser
from threading import Lock, Event

# ========================================================================================
# Data Loading for Battery Life Information and Initialization
#
# This section handles loading data from an external file for the battery life
# and initializing global variables and ROS parameters.
# ========================================================================================

file_path = 'data_the_100/matrice_battery_thrust.ods'

try:
    data = pd.read_excel(file_path, engine='odf')
except Exception as e:
    rospy.logerr(f"Failed to load battery thrust data: {e}")
    sys.exit(1)

saved_thrust = 38.0
current_battery_level = None
current_roll = 0.0
current_pitch = 0.0

state_lock = Lock()
battery_initialized_event = Event()  

# ========================================================================================
# Command-Line Arguments
#
# This section sets up the command-line argument parser and defines options for the 
# controller type, trajectory type, speed, thrust, and other parameters.
# ========================================================================================

parser = OptionParser()
parser.add_option("", "--controller", action="store", dest="controller_type", type="string", help="Controller type (FLASH, OBSTACLE, MULTI_OBSTACLES, FLASHPOINT, SUPERMAN)", default="FLASH")
parser.add_option ("", "--vicon", action="store_true", dest="vicon", help="Vicon")
parser.add_option ("", "--hover", action="store_true", dest="hover", help="Hover")

parser.add_option("-x", "", action="store", dest="x", type="float", default=0.0, help='pick a point for x')
parser.add_option("-y", "", action="store", dest="y", type="float", default=0.0, help='pick a point for y')
parser.add_option("-z", "", action="store", dest="z", type="float", default=1.0, help='pick a point for z')
parser.add_option("", "--speed", action="store", dest="speed", type="float", default=0.35, help='set the speed of the drone')

#parser.add_option ("", "--traj", action="store", dest="traj", type="int", help="Trajectory", default=3)
parser.add_option("", "--trajectory_type", action="store", dest="trajectory_type", type="string", help="Trajectory type (e.g., sinusoidal, step_z, step_xyz, spline, long_spline, rectangle, hexagon)", default="go_to_point")
parser.add_option ("", "--thrust", action="store", dest="thrust", type="int", help="Thrust", default=37)    
parser.add_option ("", "--test", action="store_true", dest="test", help="Test")    
parser.add_option("", "--obstacle", action="store", dest="rviz_obstacle", type="string", help="Rviz obstacle type (NO_OBSTACLE, OBSTACLE, MULTI_OBSTACLES)", default="NO_OBSTACLE")
parser.add_option("", "--name", action="store", dest="name", type="str", default='current',
                        help='current or full')
(options, args) = parser.parse_args()

saved_speed = options.speed

# ========================================================================================
# Controller Initialization
#
# This section initializes the appropriate MPCController class based on the selected
# controller type and sets initial parameters like speed and thrust.
# ========================================================================================

controller = None
speed_defined = False

if options.controller_type == 'FLASH':
    from py_nmpc_dji100_FLASH import MPCController
elif options.controller_type == 'OBSTACLE':
    from py_nmpc_dji100_OBSTACLE import MPCController
elif options.controller_type == 'MULTI_OBSTACLES':
    from py_nmpc_dji100_MULTI_OBSTACLES import MPCController
elif options.controller_type == 'FLASHPOINT':
    from py_nmpc_dji100_FLASHPOINT import MPCController
elif options.controller_type == 'SUPERMAN':
    from py_nmpc_dji100_SUPERMAN import MPCController
else:
    raise ValueError("Invalid controller type specified.")

speed_adjustment = options.speed  
thrust_adjustment = options.thrust  

# ========================================================================================
# Global Variables
#
# This section defines additional global variables to manage the state, timing, and other
# aspects of the controller's operation.
# ========================================================================================

auto_flag = True
timer_flag = True
old_x = None
old_y = None
old_z = None
old_roll = None
old_pitch = None
old_yaw = None
global n_samples
n_samples = 0
roll_vel = []
pitch_vel = []
yaw_vel = []
x_vel = []
y_vel = []
z_vel = []
avel_index = 0
inside_timer_callback = False
covered_traj_points = []
yref_full_points = []
covered_traj_initialized = False
covered_timestamps = []  
elapsed_time_counter = 0
# Global variables to maintain state between timer callbacks
last_exec_time = None
start_time = None
timer_counter = 0
new_controller = None
new_controller_initialised = False
target_reached = False  # Flag to indicate if the target point is reached
target_position = [options.x, options.y, options.z] # Need for the target reached part of the code

# ========================================================================================
# Callback Functions
#
# This section includes all the callback functions to handle incoming data from various
# ROS topics, such as position, attitude, velocity, battery status, and more.
# ========================================================================================

def local_pos_callback(data):
    # print(data)
    if not options.vicon:
        global state_lock
        with state_lock:
            ## rospy.loginfo(f"Before update: {controller.current_state[:3]}")
            controller.notify_position(data.point.x, data.point.y, data.point.z)
            ## rospy.loginfo(f"After update: {controller.current_state[:3]}")

def define_speed(speed):
    global speed_defined
    speed_defined = True
    return speed

def move_to_point(x, y, z):
    if hasattr(controller, 'notify_fly_to_point'):
        controller.notify_fly_to_point(x, y, z)
    else:
        rospy.logerr("Controller does not have the method 'notify_fly_to_point'")

def battery_callback(data):
    global current_battery_level, battery_initialized_event, controller, thrust_adjustment, speed_adjustment
    # Intialize the controller once
    if controller:
        return
    
    current_battery_level = data.percentage
    rospy.loginfo(f"Received battery level: {current_battery_level}%")
    thrust_adjustment = get_thrust_based_on_battery(current_battery_level)
    speed_adjustment = define_speed(options.speed)  # Set the speed before initializing the controller

    rospy.loginfo(f"Battery level: {current_battery_level}% -> Adjusting initial thrust to: {thrust_adjustment}")

    try:
        if speed_defined:
            controller = MPCController(speed_adjustment, thrust_adjustment, dt=dt, trajectory_type=options.trajectory_type)
            controller.create_nmpc_problem()
            move_to_point(options.x, options.y, options.z)
            battery_initialized_event.set()
        else:
            rospy.logwarn("Speed not defined yet, waiting for speed to be set before initializing controller.")
    except Exception as e:
        rospy.logerr(f"Failed to initialize controller: {e}")


def get_thrust_based_on_battery(battery_level):
    val = 0.0
    if battery_level in data['Battery'].values:
        valid_entries = data[data['Battery'] == battery_level]
        val = valid_entries['Thrust'].mean()
    else:
        lower = data[data['Battery'] < battery_level].max()
        upper = data[data['Battery'] > battery_level].min()
        if not lower.empty and not upper.empty:

            val = np.interp(battery_level,
                            [lower['Battery'], upper['Battery']],
                            [lower['Thrust'], upper['Thrust']])   # lienar interpolation
    if val < 38.0:
        rospy.logerr(f"Error: Abnormal value: {val}")
        val = 38.0
    if val > 44.0:
        rospy.logerr(f"Error: Abnormal value: {val}")        
        val = 44.0
    return val

def attitude_callback(data):
    try:
        # print(data)
        global n_samples
        global old_roll, old_pitch, old_yaw, roll_vel, pitch_vel, yaw_vel, avel_index
        if options.vicon:
            global current_roll, current_pitch
            (current_roll, current_pitch, current_yaw) = euler_from_quaternion([data.quaternion.x, data.quaternion.y, data.quaternion.z, data.quaternion.w])
        else:
            controller.notify_attitude(data.quaternion.x, data.quaternion.y, data.quaternion.z, data.quaternion.w)
            quat_list = [data.quaternion.x, data.quaternion.y, data.quaternion.z, data.quaternion.w]
            (roll, pitch, yaw) = euler_from_quaternion (quat_list)
            if not old_roll:
                old_roll = roll
                old_pitch = pitch
                old_yaw = yaw
                
            #rospy.loginfo(f"Before update: {controller.current_state[3:6]}")
            controller.notify_angles(roll, pitch, yaw)
            #rospy.loginfo(f"After update: {controller.current_state[3:6]}")

            n_in_average = 5
            vroll = (roll - old_roll)/dt
            vpitch = (pitch - old_pitch)/dt
            vyaw = (yaw - old_yaw)/dt
            if n_samples < n_in_average:
                #print("SAVERPY:", n_samples)
                roll_vel.append(vroll)
                pitch_vel.append(vpitch)
                yaw_vel.append(vyaw)
                n_samples += 1
            else:
                #print("avel_index:", avel_index)
                roll_vel[avel_index] = vroll
                pitch_vel[avel_index] = vpitch
                yaw_vel[avel_index] = vyaw
                avel_index += 1
                avel_index %= n_in_average
            rsum = 0.0
            psum = 0.0
            ysum = 0.0
            # print("N_SAMPLES:", n_samples)
            for i in range(n_samples):
                rsum += roll_vel[i];
                psum += pitch_vel[i];
                ysum += yaw_vel[i];
                # print("SUMS:", rsum, psum, ysum)
            controller.notify_angle_rates(rsum/n_samples, psum/n_samples, ysum/n_samples);
            old_roll = roll
            old_pitch = pitch
            old_yaw = yaw
    except Exception as e:
        print(f"Exception occurred: {e}")
    
def velocity_callback(data):
    # print(data)
    if not options.vicon:
        controller.notify_velocity(data.vector.x, data.vector.y, data.vector.z)

def speed_callback(data):
    global saved_speed
    saved_speed = data.data

def target_callback(data):
    global new_controller, new_controller_initialised, saved_thrust, saved_speed, target_position, target_reached
    print("NEW TARGET:", dt, options.trajectory_type, data, saved_thrust, saved_speed)
    new_controller = MPCController(saved_speed, saved_thrust, dt=dt, trajectory_type="go_to_point")
    # new_controller = MPCController(speed_adjustment, thrust_adjustment, dt=dt, trajectory_type=options.trajectory_type)
    new_controller.create_nmpc_problem()  # Make sure this is called here
    new_controller.notify_fly_to_point(data.x, data.y, data.z)
    target_position = [data.x, data.y, data.z]
    target_reached = False    
    new_controller_initialised = True

def target_speed_callback(data):
    global new_controller, new_controller_initialised, saved_thrust, target_position, target_reached
    print("NEW TARGET:", dt, options.trajectory_type, data, saved_thrust, data.w)
    new_controller = MPCController(data.w, saved_thrust, dt=dt, trajectory_type=options.trajectory_type)
    # new_controller = MPCController(speed_adjustment, thrust_adjustment, dt=dt, trajectory_type=options.trajectory_type)
    new_controller.create_nmpc_problem()  # Make sure this is called here
    new_controller.notify_fly_to_point(data.x, data.y, data.z)
    target_position = [data.x, data.y, data.z]
    target_reached = False
    new_controller_initialised = True

def hover_callback(data):
    basic_hover()

def basic_hover():
    rospy.logerr("Entering hover mode...")
    global new_controller, new_controller_initialised, last_u_opt, controller
    # Initialize a new controller for hover
    new_controller = MPCController(saved_speed, saved_thrust, dt=dt, trajectory_type='hover')
    new_controller.create_nmpc_problem()
    # Set the current position as the hover point
    new_controller.notify_fly_to_point(controller.current_state[0], controller.current_state[1], controller.current_state[2])
    new_controller_initialised = True

def rc_callback(data):
    # print(data)
    global auto_flag
    old_auto_flag = auto_flag
    auto_flag = data.axes[4] > 5000.0
    if old_auto_flag != auto_flag:
        print("AUTO FLAG CHANGED TO:", auto_flag)
    
'''     
def hokuyo_lidar_callback(data):
    # print(data)
    if not options.vicon:
        controller.notify_obstacle(data.point.radius, data.point.center)
'''

def tf_callback(data):
    ## print(data)
    global old_x, old_y, old_z, n_samples, avel_index
    if options.vicon:
        for trans in data.transforms:
            if trans.header.frame_id == "world" and trans.child_frame_id == "mat2":
                vicon_dt = 0.02
                x = trans.transform.translation.x
                y = trans.transform.translation.y
                z = trans.transform.translation.z
                current_x = x
                current_y = y
                current_z = z

                # print("*********************************************************************************")
                controller.notify_position(x, y, z)

                if not old_x:
                    old_x = x
                    old_y = y
                    old_z = z

                vel_n_in_average = 5
                vx = (current_x - old_x)/vicon_dt
                vy = (current_y - old_y)/vicon_dt
                vz = (current_z - old_z)/vicon_dt
                if n_samples < vel_n_in_average:
                    x_vel.append(vx)
                    y_vel.append(vy)
                    z_vel.append(vz)
                    n_samples += 1
                else:
                    #print("UPDATE:", avel_index, vx, vy, vz)
                    x_vel[avel_index] = vx
                    y_vel[avel_index] = vy
                    z_vel[avel_index] = vz
                    avel_index += 1
                    avel_index %= vel_n_in_average
                vx_sum = 0.0
                vy_sum = 0.0
                vz_sum = 0.0
                for i in range(n_samples):
                    vx_sum += x_vel[i];
                    vy_sum += y_vel[i];
                    vz_sum += z_vel[i];
                    # print("VSUM;", i, vx_sum, vy_sum, vz_sum)
                controller.notify_velocity(vx_sum/n_samples, vy_sum/n_samples, vz_sum/n_samples)

                msg = Vector3();
                msg.x = vx;
                msg.y = vy;
                msg.z = vz;

                velocity_pub.publish(msg);

                qx = trans.transform.rotation.x
                qy = trans.transform.rotation.y
                qz = trans.transform.rotation.z
                qw = trans.transform.rotation.w

                global current_roll, current_pitch
                (roll, pitch, current_yaw) = euler_from_quaternion([qx, qy, qz, qw])

                controller.notify_angles(current_roll, current_pitch, current_yaw);     
                old_x = x
                old_y = y
                old_z = z

# ========================================================================================
# Visualization Functions
#
# This section includes functions to create and publish various markers in RViz for 
# visualizing the trajectory, obstacles, and other relevant information.
# ========================================================================================

def publish_room_boundaries(marker_pub):
    # Define the room dimensions
    width = 2.5
    depth = 2.5
    height = 3.0
    half_width = width / 2
    half_depth = depth / 2
    half_height = height / 2
    lift_z = height/2 # offset

    corners = [
        Point(-half_width, -half_depth, -half_height + lift_z),  # Bottom corners
        Point(half_width, -half_depth, -half_height + lift_z),
        Point(half_width, half_depth, -half_height + lift_z),
        Point(-half_width, half_depth, -half_height + lift_z),
        Point(-half_width, -half_depth, half_height + lift_z),  # Top corners
        Point(half_width, -half_depth, half_height + lift_z),
        Point(half_width, half_depth, half_height + lift_z),
        Point(-half_width, half_depth, half_height + lift_z)
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Create a marker for the edges
    edge_marker = Marker()
    edge_marker.header.frame_id = "world"
    edge_marker.header.stamp = rospy.Time.now()
    edge_marker.ns = "room_edges"
    edge_marker.id = 0
    edge_marker.type = Marker.LINE_LIST
    edge_marker.action = Marker.ADD
    edge_marker.pose.orientation.w = 1.0
    edge_marker.scale.x = 0.02  # Line width
    edge_marker.color.a = 1.0  
    edge_marker.color.r = 1.0  # Red
    edge_marker.color.g = 1.0  # Green
    edge_marker.color.b = 0.0  # Blue

    # Add points to the marker
    for start, end in edges:
        edge_marker.points.append(corners[start])
        edge_marker.points.append(corners[end])

    # Publish the marker
    marker_pub.publish(edge_marker)

def publish_timestamps(marker_pub, elapsed_time_counter, points, timestamps, ns, id_offset, name='name'):

    if len(points) != len(timestamps):
        rospy.logerr("Error: Points and timestamps list must have the same length.")
        return
    
    ma = MarkerArray()
    for i, (point, timestamp) in enumerate(zip(points, timestamps)):
 
        if name == 'full':
            if (i % 10) != 0:
                continue

        #print("Inside the for loop", name, i, timestamp)
        # Cube Marker
        cube_marker = Marker()
        cube_marker.header.frame_id = "world"
        cube_marker.header.stamp = rospy.Time.now()
        cube_marker.ns = ns + "_cubes"
        cube_marker.id = id_offset + i * 2
        cube_marker.type = Marker.CUBE
        cube_marker.action = Marker.ADD
        cube_marker.pose.position = point
        cube_marker.scale.x = cube_marker.scale.y = cube_marker.scale.z = 0.1  # Set cube size
        cube_marker.color.a = 1.0  # Don't forget to set the alpha!
        if name == 'full':
            cube_marker.color.r = 1.0
            cube_marker.color.g = 0.0
            cube_marker.color.b = 0.0
        elif name == 'current':
            cube_marker.color.r = 165 / 255.0
            cube_marker.color.g = 42 / 255.0
            cube_marker.color.b = 42 / 255.0

        # Text Marker
        text_marker = Marker()
        text_marker.header.frame_id = "world"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = ns + "_texts"
        text_marker.id = id_offset + i * 2 + 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = point.x
        text_marker.pose.position.y = point.y
        if name == 'full':
            text_marker.pose.position.z = point.z - 0.2  # Raise the text a bit below the cube
        elif name == 'current':
            text_marker.pose.position.z = point.z + 0.2  # Raise the text a bit above the cube
        text_marker.scale.z = 0.2  # Text size
        text_marker.color.a = 1.0
        text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0  # White text  
        if name == 'full':
            text_marker.text = f"{timestamp:.2f}"
        elif name == 'current':
            text_marker.text = f"{elapsed_time_counter:.2f}"

        # Publish both markers
        #marker_pub.publish(cube_marker)
        #marker_pub.publish(text_marker)
        ma.markers.append(cube_marker)
        ma.markers.append(text_marker)
    marker_array_pub.publish(ma)

def publish_obstacle_as_sphere(marker_pub, obstacle_center, obstacle_radius, ns="obstacle", id=0):
    if obstacle_center is None or obstacle_radius is None or ns is None:
        return

    obstacle_marker = Marker()
    obstacle_marker.header.frame_id = "world"
    obstacle_marker.header.stamp = rospy.Time.now()
    obstacle_marker.ns = ns
    obstacle_marker.id = id
    obstacle_marker.type = Marker.SPHERE
    obstacle_marker.action = Marker.ADD
    obstacle_marker.pose.position.x = obstacle_center[0]
    obstacle_marker.pose.position.y = obstacle_center[1]
    obstacle_marker.pose.position.z = obstacle_center[2]
    obstacle_marker.scale.x = obstacle_radius * 2  # Diameter in X
    obstacle_marker.scale.y = obstacle_radius * 2  # Diameter in Y
    obstacle_marker.scale.z = obstacle_radius * 2  # Diameter in Z
    obstacle_marker.color.a = 1.0  # Alpha transparency
    obstacle_marker.color.r = 1.0
    obstacle_marker.color.g = 0.7
    obstacle_marker.color.b = 0.0

    # Publish the marker
    marker_pub.publish(obstacle_marker)

def publish_prediction_horizon_with_arrows(marker_pub, prediction_horizon, ns, id_offset):
    for i in range(prediction_horizon.shape[1] - 1):
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "world"
        arrow_marker.header.stamp = rospy.Time.now()
        arrow_marker.ns = ns + "_arrows"
        arrow_marker.id = id_offset + i
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.points.append(Point(x=prediction_horizon[0, i], y=prediction_horizon[1, i], z=prediction_horizon[2, i]))
        arrow_marker.points.append(Point(x=prediction_horizon[0, i + 1], y=prediction_horizon[1, i + 1], z=prediction_horizon[2, i + 1]))
        arrow_marker.scale.x = 0.05  # Shaft diameter
        arrow_marker.scale.y = 0.1   # Head diameter
        arrow_marker.scale.z = 0     # Head length, not used for arrows
        arrow_marker.color.a = 1.0   
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.0
        arrow_marker.color.b = 0.0
        
        marker_pub.publish(arrow_marker)
 
def display_green_thingy(yref_full_points, marker_pub, traj_id=0):
    # Create a new marker for the full reference trajectory
    traj_marker = Marker()
    traj_marker.header.frame_id = "world"
    traj_marker.header.stamp = rospy.Time.now()
    traj_marker.ns = "full_reference_trajectory"
    traj_marker.id = traj_id
    traj_marker.type = Marker.LINE_STRIP
    traj_marker.action = Marker.ADD
    traj_marker.scale.x = 0.05  # Line width
    traj_marker.color.a = 1.0   
    traj_marker.color.r = 0.0   
    traj_marker.color.g = 1.0   # Green
    traj_marker.color.b = 0.0  

    traj_marker.points.extend(yref_full_points) # The full green one published once

    marker_pub.publish(traj_marker)

    num_points = len(yref_full_points)
    timestamps = [i * 0.1 for i in range(num_points)]  

    #rospy.logerr("LOOOK HEREE:")
    #print("Yref Full points", yref_full_points)
    #print("Timestamps", timestamps)

    # Publish timestamps for the full trajectory
    publish_timestamps(marker_pub, elapsed_time_counter, yref_full_points, timestamps, ns, 4000, name='full')

    # FIX OBSTACLES LATER TOO

def vidi_sta_se_desava(marker_pub, covered_traj_points, yref_traj_points, obstacle_center=None, obstacle_radius=None, obstacles=[], covered_id=1, yref_id=3, obstacle_id=2, obstacle_start_id=100):
    
    # Initialize the LINE_STRIP marker for the trajectory covered so far
    covered_traj_marker = Marker()
    covered_traj_marker.header.frame_id = "world"
    covered_traj_marker.header.stamp = rospy.Time.now()
    covered_traj_marker.ns = "covered_trajectory"
    covered_traj_marker.id = covered_id
    covered_traj_marker.type = Marker.LINE_STRIP
    covered_traj_marker.action = Marker.ADD
    covered_traj_marker.scale.x = 0.05  # Line width
    covered_traj_marker.color.a = 1.0
    covered_traj_marker.color.r = 0.0  # Blue
    covered_traj_marker.color.g = 0.0
    covered_traj_marker.color.b = 1.0

    covered_traj_marker.points.extend(covered_traj_points)

    # Initialize the LINE_STRIP marker for yref trajectory
    yref_traj_marker = Marker()
    yref_traj_marker.header.frame_id = "world"
    yref_traj_marker.header.stamp = rospy.Time.now()
    yref_traj_marker.ns = "reference trajectory"
    yref_traj_marker.id = yref_id
    yref_traj_marker.type = Marker.LINE_STRIP
    yref_traj_marker.action = Marker.ADD
    yref_traj_marker.scale.x = 0.19  # Line width
    yref_traj_marker.color.a = 1.0
    yref_traj_marker.color.r = 1.0  # Yellow
    yref_traj_marker.color.g = 0.7
    yref_traj_marker.color.b = 0.0

    ps = PointStamped()
    ps.header.frame_id = "world"
    ps.header.stamp = rospy.Time.now()
    ps.point = yref_traj_points[0]
    first_yellow_pub.publish(ps)

    yref_traj_marker.points.extend(yref_traj_points)

    # Publish the markers
    marker_pub.publish(covered_traj_marker)
    marker_pub.publish(yref_traj_marker)   

    publish_obstacle_as_sphere(marker_pub, obstacle_center, obstacle_radius, ns="obstacle", id=obstacle_id)
    #rospy.logerr("Sta se ovde desava?")
    #rospy.logerr("Ne zaista sta se ovde desava?")
    for idx, obstacle in enumerate(obstacles):
        obstacle_center = obstacle['center']
        obstacle_radius = obstacle['radius']
        publish_obstacle_as_sphere(marker_pub, obstacle_center, obstacle_radius, ns="obstacle", id=obstacle_start_id + idx)
    
    yref_traj_points.clear() 

def initialize_trajectories(drone_initial_position, initial_ref_offset):
    global yref_full_points, covered_traj_points, yref_traj_points

    initial_point = Point(x=drone_initial_position[0], y=drone_initial_position[1], z=drone_initial_position[2])
    initial_point_ref = Point(x=initial_ref_offset[0], y=initial_ref_offset[1], z=initial_ref_offset[2])
    yref_full_points = [initial_point_ref]  
    covered_traj_points = [initial_point]  # Actual or covered trajectory (Blue)
    yref_traj_points = [initial_point_ref] 

def update_trajectory_with_offset(drone_current_position, yref_full_original):
    #offset = np.array(drone_current_position) + np.array(yref_full_original[:, 0])
    yref_full_adjusted = yref_full_original #+ offset[:, None]  # Apply offset to all points
    global yref_full_points
    yref_full_points = []  # Reset yref_full_points to ensure it's updated correctly
    for col in range(yref_full_adjusted.shape[1]):
        # Create Point objects for each position
        point = Point(x=float(yref_full_adjusted[0, col]), y=float(yref_full_adjusted[1, col]), z=float(yref_full_adjusted[2, col]))
        yref_full_points.append(point)

def append_current_position_to_covered_trajectory(drone_current_position):
    global covered_traj_points
    current_point = Point(x=drone_current_position[0], y=drone_current_position[1], z=drone_current_position[2])
    covered_traj_points.append(current_point)

def append_current_position_to_yref_trajectory(yref):
    global yref_traj_points
    # Loop through the yref_data in steps of 10 to extract each x, y, z coordinate
    for i in range(0, len(yref), 10):
        x, y, z = yref[i], yref[i+1], yref[i+2]  # Extract x, y, z coordinates
        yref_point = Point(x=x, y=y, z=z)  # Create a Point object
        yref_traj_points.append(yref_point)  # Append the point to the trajectory

def append_current_position_timestamp(drone_current_position, elapsed_time_counter):
    global covered_timestamps, covered_traj_points

    current_time = rospy.Time.now().to_sec()  # Ensure you capture the current time at the function start
    current_point = Point(x=drone_current_position[0], y=drone_current_position[1], z=drone_current_position[2])
    covered_timestamps.append(current_time)  # Storing the current time as the timestamp
    
    publish_timestamps(marker_pub, elapsed_time_counter, [current_point], [current_time], "covered_timestamps", 2000 + len(covered_timestamps), name='current')

def check_if_near_target(current_state, target_state, tolerance=0.5):
    print("check_if_near_target CURRENT STATE - TARGETSTATE:" , current_state, target_state)
    return np.linalg.norm(np.array(current_state) - np.array(target_state)) < tolerance

"""
# THE SPOT WHERE EVERYTHING STARTS connected to the controller FLASH / OBSTACLE / MULTI_OBSTACLES
"""
# ========================================================================================
# TIMER CALLBACK FUNCTION
#
# This section includes functions to create and publish various markers in RViz for 
# visualizing the trajectory, obstacles, and other relevant information.
# ========================================================================================

def timer_callback(event): 
    print("timer_callback")

    global timer_counter, controller, covered_traj_initialized, new_controller, new_controller_initialised, in_hover_flag

    if new_controller and new_controller_initialised:
        covered_traj_initialized = False        
        controller = new_controller
        new_controller = None
        new_controller_initialised = False

    if not controller:
        return

    if not controller.start_point and (options.trajectory_type in ["hover", "go_to_point"]):
        return

    if not controller.have_target_position and options.trajectory_type in ["go_to_point"]:
        return

    #if options.trajectory_type in ["hover"]:
    #    print("HOVER INIT*****************:", controller.start_point)
    #    basic_hover()
    #    options.trajectory_type = ""
        
    if controller.start_point and controller.have_target_position and not controller.init_speed_flag:
        rospy.logerr("POGLEDAJ OVO DA LI ULAZI SVAKI PUT U INIT_SPEED?")
        controller.init_speed()

    #print("****")
    
    timer_counter += 1
    if timer_counter == 100 and False:
        covered_traj_initialized = False
        controller = MPCController(speed_adjustment, thrust_adjustment, dt=dt, trajectory_type=options.trajectory_type)
        controller.create_nmpc_problem()  # Make sure this is called here
        move_to_point(-options.x, -options.y, options.z)
        define_speed(options.speed)

    global timer_flag, last_exec_time, start_time, elapsed_time_counter, state_lock
    global inside_timer_callback, yref_full_original, yref_traj_points

    #rospy.loginfo("Current state at timer callback: {}".format(controller.current_state))

    if controller.have_position:
        yref_full, dt_full, Duration_full = controller.full_trajectory_rviz(trajectory_type=options.trajectory_type)
    
    current_time = rospy.Time.now()

    #print("****")
    
    if start_time is None:
        start_time = current_time  # Initialize start time
        last_exec_time = current_time  # And set the last execution time

    # Calculate the time since the last execution
    time_since_last_exec = (current_time - last_exec_time).to_sec()

    # Update elapsed time counter with the time since last execution
    elapsed_time_counter += time_since_last_exec

    if not auto_flag:
        print("IN MANUAL: Do nothing")
        return

    if not timer_flag:
        print("Time is disabled")
        return

    if inside_timer_callback:
        print("ERROR: CONTROLLER NOT FAST ENOUGH")
        return
    
    inside_timer_callback = True

    if options.test:
        print("TEST THRUST:", options.thrust)
        msg = Joy()
        msg.axes.append(0) # roll
        msg.axes.append(0) # pitch
        msg.axes.append(options.thrust) # thrust
        msg.axes.append(0) # yaw_rate
        msg.axes.append(0x02 | 0x01 | 0x08 | 0x20)
        ctrl_pub.publish(msg)
        return

    # ----------------------------------------------------------------
    # TOMMY - ORIGINAL
    #u_opt, yref = controller.tick(traj=options.traj)
    ###print("YREF:", yref.T)
    #ref_state = yref.T[0]

    ### print("U_opt:", u_opt)
    #print(f'Control {controller.current_time:.2f}: {u_opt[0]:.2f} {math.degrees(u_opt[1]):.2f} {math.degrees(u_opt[2]):.2f} {math.degrees(u_opt[3]):.2f} - {ref_state[0]} {ref_state[1]} {ref_state[2]} - {ref_state[6]} {ref_state[7]} {ref_state[8]}')
    #if u_opt[0]:
    #    msg = Joy()
    #    msg.axes.append(-u_opt[1]) # roll
    #    msg.axes.append(-u_opt[2]) # pitch
    #    msg.axes.append(u_opt[0]*7.0/14.6+37.11) # thrust
    #    msg.axes.append(u_opt[3]) # yaw_rate
    #    msg.axes.append(0x02 | 0x01 | 0x08 | 0x20)
    #    ctrl_pub.publish(msg)
    # ----------------------------------------------------------------
    
    # ----------------------------------------------------------------
    # LARA - CHANGE
    with state_lock:
        u_opt, yref, xk, states = controller.tick(controller.current_time, trajectory_type=options.trajectory_type)

    #rospy.logerr("POGLEDAJ OVO JAKO JE BITNO xk kako izgleda")
    #print("sta se desava", xk[:3])
    #print("YREF see here:", yref)
    ref_state = yref
    #rospy.logerr("The yref trajectory that needs to be printed yellow: %s" % yref)

    initial_ref_offset = np.array(xk[:3]) + np.array(ref_state[:3])
    #print("Initial reference position with ofset:", initial_ref_offset)
 
    #print("Print the prediction horizon", states[:3])
    #print("ref_state", ref_state[:3]) # --- so it plots everything once only
                                      # so that the yellow line and blue start from the same position
    #print("current_state", xk[:3])
    #print("current_state FULL", xk)

    if not covered_traj_initialized:
        yref_full_original = yref_full.copy()  # Preserve the original trajectory
        controller.current_state[:3] = [0, 0, 0]  # Resets only the current position once
        initialize_trajectories(xk[:3], initial_ref_offset)  # Set initial positions for both trajectories
        update_trajectory_with_offset(xk[:3], yref_full_original)  # Adjust the green trajectory only once
        covered_traj_initialized = True
        #display_green_thingy(yref_full_points, marker_pub, traj_id=0)
        #timer_flag = False
    publish_room_boundaries(marker_pub)
    display_green_thingy(yref_full_points, marker_pub, traj_id=0)

    #rospy.logerr("STA SE DESAVA: {:.2f}".format((int(elapsed_time_counter) % 10)) )
    print("KOJA JE BRZINA", controller.speed)
    #rospy.logerr("STA SE DESAVA svasta se desava: {:.2f}".format((elapsed_time_counter) ))

    # Check if the elapsed_time_counter is close to an integer
    if math.isclose(elapsed_time_counter, round(elapsed_time_counter), abs_tol=1e-2):
        print("Elapsed Time Counter Updated to: {:.2f}".format(elapsed_time_counter))
        append_current_position_timestamp(xk[:3], elapsed_time_counter)
 
    append_current_position_to_covered_trajectory(xk[:3])  # Blue trajectory    
    append_current_position_to_yref_trajectory(yref)  # Yellow trajectory 

    publish_prediction_horizon_with_arrows(marker_pub, states[:3], "prediction_horizon", 3000)  # Red arrows prediction horizon

    print("The optimal control is u_opt:", u_opt)
    
    print(f'Control {controller.current_time:.2f}: {u_opt[0]:.2f} {math.degrees(u_opt[1]):.2f} {math.degrees(u_opt[2]):.2f} {math.degrees(u_opt[3]):.2f} - {ref_state[0]} {ref_state[1]} {ref_state[2]} - {ref_state[6]} {ref_state[7]} {ref_state[8]}')
 
    global target_reached

    if controller.trajectory_type == "go_to_point" and check_if_near_target(xk[:3], target_position) and False:
        if not target_reached:
            target_reached = True
            # last_u_opt = u_opt
            #rospy.loginfo("Target reached. Hovering in place.")
            rospy.logerr("Target reached. Hovering in place.")
            basic_hover()  # Call the hover callback to switch to hover mode

    '''
    if target_reached:
        msg = Joy()
        msg.axes.append(last_u_opt[1])  # roll
        msg.axes.append(last_u_opt[2])  # pitch
        msg.axes.append(thrust_adjustment)  # thrust
        msg.axes.append(last_u_opt[3])  # yaw_rate
        msg.axes.append(0x02 | 0x01 | 0x08 | 0x20)
    else:
    '''
    if u_opt[0]:
        msg = Joy()
        msg.axes.append(u_opt[1]) # roll
        msg.axes.append(u_opt[2]) # pitch
        msg.axes.append(u_opt[0]) # thrust
        msg.axes.append(u_opt[3]) # yaw_rate
        msg.axes.append(0x02 | 0x01 | 0x08 | 0x20)
        roll = Float64()
        roll.data = math.degrees(u_opt[1])
        pitch = Float64()
        pitch.data = math.degrees(u_opt[2])
        thrust = Float64()
        thrust.data = u_opt[0]
        yaw_rate = Float64()
        yaw_rate.data = math.degrees(u_opt[3])
        ctrl_roll_pub.publish(roll)
        ctrl_pitch_pub.publish(pitch)
        ctrl_thrust_pub.publish(thrust)
        ctrl_yaw_rate_pub.publish(yaw_rate)
        ctrl_pub.publish(msg)
        global saved_thrust
        saved_thrust = thrust.data
    # ----------------------------------------------------------------

    inside_timer_callback = False

    elapsed_time = (current_time - start_time).to_sec()  # Total elapsed time since start

    print("CONTROL DURATION:", elapsed_time)

    print("CONTROL MODE:", controller.trajectory_type)

    # Update last_exec_time to current time after operations
    last_exec_time = current_time

    #print("Elapsed Time Counter Updated to: {:.2f}".format(elapsed_time_counter))
    #print("Time since last execution: {:.2f} seconds".format(time_since_last_exec))

    if options.rviz_obstacle == 'OBSTACLE':
        vidi_sta_se_desava(marker_pub, covered_traj_points, yref_traj_points, controller.obstacle_center, controller.obstacle_radius, covered_id=1, yref_id=3, obstacle_id=2, obstacle_start_id=100)
    elif options.rviz_obstacle == 'MULTI_OBSTACLES':
        vidi_sta_se_desava(marker_pub, covered_traj_points, yref_traj_points, controller.obstacle_center, controller.obstacle_radius, controller.obstacles, covered_id=1, yref_id=3, obstacle_id=2, obstacle_start_id=100)
    else:
        vidi_sta_se_desava(marker_pub, covered_traj_points, yref_traj_points, covered_id=1, yref_id=3, obstacle_id=2, obstacle_start_id=100)

# ========================================================================================
# Main Script Execution
#
# This section initializes the ROS node, sets up the publishers and subscribers, and starts
# the main loop to keep the node running.
# ========================================================================================

if __name__ == "__main__":
    rospy.init_node ("pycontroller")
    ns = rospy.get_namespace ().rstrip("/")

    battery_sub = rospy.Subscriber(ns + "/dji_sdk/battery_state", BatteryState, battery_callback)
    dt = 0.1

    rospy.loginfo("Waiting for the battery level and initializing the controller...")
    battery_initialized_event.wait(timeout=10)  # timeout = 10
    if not battery_initialized_event.is_set():
        rospy.logerr("Timeout waiting for battery data.")
        sys.exit(1)

    # Publishers
    marker_array_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, latch=False, queue_size=1000)
    marker_pub = rospy.Publisher("/visualization_marker", Marker, latch=False, queue_size=1000)   # queue_size ....
    ctrl_pub = rospy.Publisher("dji_sdk/flight_control_setpoint_generic", Joy, latch=False, queue_size=10)
    ref_pos_pub = rospy.Publisher("tune/ref_position", PointStamped, latch=False, queue_size=10)
    first_yellow_pub = rospy.Publisher("tune/first_yellow", PointStamped, latch=False, queue_size=10)
    err_x_pub = rospy.Publisher("tune/err_x", Float64, latch=False, queue_size=10)
    err_y_pub = rospy.Publisher("tune/err_y", Float64, latch=False, queue_size=10)
    err_z_pub = rospy.Publisher("tune/err_z", Float64, latch=False, queue_size=10)
    ctrl_roll_pub = rospy.Publisher("ctrl/roll", Float64, latch=False, queue_size=10)
    ctrl_pitch_pub = rospy.Publisher("ctrl/pitch", Float64, latch=False, queue_size=10)
    ctrl_thrust_pub = rospy.Publisher("ctrl/thrust", Float64, latch=False, queue_size=10)
    ctrl_yaw_rate_pub = rospy.Publisher("ctrl/yaw_rate", Float64, latch=False, queue_size=10)
    velocity_pub = rospy.Publisher("velocity", Vector3, latch=False, queue_size=10)
    #hokuyo_lidar_pub = rospy.Publisher("hokuyo_scan", LaserScan, latch=False, queue_size=10)

    # Subscribers
    local_pos_sub = rospy.Subscriber("world_position", PointStamped, local_pos_callback)       #/dji_sdk/local_position
    attitude_sub = rospy.Subscriber("dji_sdk/attitude", QuaternionStamped, attitude_callback)
    velocity_sub = rospy.Subscriber("dji_sdk/velocity", Vector3Stamped, velocity_callback)
    target_sub = rospy.Subscriber("target", Vector3, target_callback)
    speed_sub = rospy.Subscriber("speed", Float64, speed_callback)
    target_speed_sub = rospy.Subscriber("target_speed", Quaternion, target_speed_callback)
    hover_sub = rospy.Subscriber("hover", Empty, hover_callback)

    rc_sub = rospy.Subscriber("dji_sdk/rc", Joy, rc_callback)
    #hokuyo_lidar_sub = rospy.Subscriber("/scan", LaserScan, hokuyo_lidar_callback)

    tf_sub = rospy.Subscriber("/tf", TFMessage, tf_callback)

    # Timer for control loop
    rospy.Timer(rospy.Duration(dt), timer_callback)
    
    print("Spinning pycontroller node")
    
    rospy.spin()
    
