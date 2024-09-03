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

'''
CODE FOCUSED JUST ON THE LIDAR SCANNING DATA INFORMATION 
'''

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float64
from visualization_msgs.msg import Marker
import math

# Global variables for subscribed topics
hokuyo_scan = None  # New variable for Lidar measurements
hasLaserScanData = False

# Constants for PI controller
desired_horizontal_distance = 5.32  # Desired horizontal distance from the wall
centeraverageRange = 0.0

# Publishers
average_distance_pub = None
hokuyo_scan_publisher = None
control_signal_pub = None
error_pub = None
marker_pub = None  # Publisher for RViz markers

def processHokuyoSensorData():
    global centeraverageRange
    if not hasLaserScanData:
        rospy.logwarn("No laser scan data available.")
        return 0.0

    # Calculate the control signal based on the PI controller
    error = centeraverageRange - desired_horizontal_distance
    rospy.loginfo(f"Horizontal avg distance to the wall at the moment: {centeraverageRange} meters")
    rospy.loginfo(f"Error: {error} meters")

    # Call the MPC controller function to get the control input
    # Replace with actual MPC computation
    control_signal = computeMPCControl(centeraverageRange)

    rospy.loginfo(f"Control_signal: {control_signal} m/s")

    # Publish control signal for rqt_plot
    control_signal_msg = Float64()
    control_signal_msg.data = control_signal
    control_signal_pub.publish(control_signal_msg)

    error_msg = Float64()
    error_msg.data = error
    error_pub.publish(error_msg)

    return control_signal


def hokuyoScanCallback(msg):
    global hokuyo_scan, hasLaserScanData, centeraverageRange
    hokuyo_scan = msg
    hasLaserScanData = True

    rospy.loginfo("Received new Lidar scan data")

    # Find the index of the center beam
    centerIndex = len(hokuyo_scan.ranges) // 2

    # Define the angle range for averaging (45 degrees on each side)
    angle_range = 16.0 * (math.pi / 180.0)  # 16 degrees in radians

    # Calculate the number of beams within the angle range
    num_beams_within_range = int(math.ceil(angle_range / hokuyo_scan.angle_increment))

    # Calculate the indices of the beams at the edges of the angle range
    left_index = max(centerIndex - num_beams_within_range, 0)
    right_index = min(centerIndex + num_beams_within_range, len(hokuyo_scan.ranges) - 1)

    # Calculate the sum of distances within the angle range
    sum_distances = sum(hokuyo_scan.ranges[left_index:right_index + 1])

    # Calculate the average distance within the angle range
    average_distance = sum_distances / (right_index - left_index + 1)

    # Get the range value of the center beam
    centeraverageRange = average_distance

    # Check if average_distance is less than 0.05 and set it to 6.5 meters
    if centeraverageRange < 0.05:
        centeraverageRange = 6.5

    rospy.loginfo(f"Average horizontal distance to the wall: {average_distance} meters")

    # Publish the average distance
    average_distance_msg = Float32()
    average_distance_msg.data = average_distance
    average_distance_pub.publish(average_distance_msg)

    # Publish the received laser scan data for visualization in RViz
    hokuyo_scan.header.stamp = rospy.Time.now()
    hokuyo_scan.header.frame_id = "world"  # Changed frame to an existing one
    hokuyo_scan_publisher.publish(hokuyo_scan)

    # Publish the distance as a dot marker in RViz
    marker = Marker()
    marker.header.frame_id = "world"  # Make sure this frame exists
    marker.header.stamp = rospy.Time.now()
    marker.ns = "lidar_distance"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = 0 #centeraverageRange
    marker.pose.position.y = centeraverageRange
    marker.pose.position.z = 0  # Adjust height as needed
    marker.scale.x = 0.81  # Dot size
    marker.scale.y = 0.81  # Dot size
    marker.scale.z = 0.81  # Dot size
    marker.color.a = 1.0  # Alpha
    marker.color.r = 0.0  # Red
    marker.color.g = 1.0  # Green
    marker.color.b = 0.0  # Blue

    marker_pub.publish(marker)

def computeMPCControl(centeraverageRange):
    # Placeholder function for MPC computation
    # Replace with actual MPC logic
    return centeraverageRange * 0.1  # Example: Proportional control


if __name__ == "__main__":
    rospy.init_node("demo_local_position_control_node")

    # Initialize publishers
    average_distance_pub = rospy.Publisher("average_distance", Float32, queue_size=10)
    hokuyo_scan_publisher = rospy.Publisher("hokuyo_scan", LaserScan, queue_size=10)
    control_signal_pub = rospy.Publisher("control_signal", Float64, queue_size=10)
    error_pub = rospy.Publisher("error_signal", Float64, queue_size=10)
    marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=1000)  # Initialize marker publisher
    # Subscribe to the hokuyo lidar topic
    hokuyo_scan_subscriber = rospy.Subscriber("/scan", LaserScan, hokuyoScanCallback)

    # Wait until laser scan data is received
    while not hasLaserScanData:
        rospy.sleep(0.1)
        rospy.loginfo("Waiting for Lidar data...")

    rospy.loginfo("Lidar data received. Entering main loop.")
    rospy.spin()
