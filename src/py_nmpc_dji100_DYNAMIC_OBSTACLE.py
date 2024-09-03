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

import math
import numpy as np
#import mpctools as mpc
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
from casadi import *
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from optparse import OptionParser
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
import argparse
import rospy
import time
import colorama
from colorama import Fore, Style

#python3 py_nmpc_dji100_MULTI_OBSTACLES.py --trajectory_type long_spline --use_nmpc

class MPCController:
    def __init__(self, initial_speed, initial_thrust, dt=0.1, trajectory_type="go_to_point", use_nmpc="False"): # dt = T/N # Include trajectory_type parameter with default value
        self.n_states = 10
        self.n_controls = 4
        self.T = 6 #6 # Time horizon
        self.N = int(self.T/dt) # Number of control intervals
        self.trajectory_type = trajectory_type  # Store trajectory type
        self.use_nmpc = use_nmpc  # NMPC usage flag
        self.start_point = None
        # Log trajectory settings
        print(f"Initializing MPCController with trajectory type: {trajectory_type}")
        print(f"Use NMPC: {use_nmpc}")
        self.x_new = MX.sym('x_new')      # states
        self.y = MX.sym('y') 
        self.z = MX.sym('z') 
        self.phi = MX.sym('phi') 
        self.theta = MX.sym('theta') 
        self.psi = MX.sym('psi') 
        self.xdot = MX.sym('zdot') 
        self.ydot = MX.sym('ydot') 
        self.zdot = MX.sym('zdot') 
        self.psidot = MX.sym('psidot') 
        self.x1 = vertcat(self.x_new, self.y, self.z, self.phi, self.theta, self.psi, self.xdot, self.ydot, self.zdot, self.psidot) # states in a column
        self.u1 = MX.sym('u1')       # control input
        self.u2 = MX.sym('u2')       # control input
        self.u3 = MX.sym('u3')       # control input
        self.u4 = MX.sym('u4')       # control input
        self.u = vertcat(self.u1, self.u2, self.u3, self.u4)      # control input

        self.x_desired_param = MX.sym('x_desired_param', self.n_states)  
        self.x_param = MX.sym('x_param', self.n_states) 
        self.u_param = MX.sym('u_param', self.n_controls) 

        self.dt = dt  # Time step
        self.mass = 2.895
        self.gravity = 9.81
        self.mg = self.mass * self.gravity  # mass times gravity
        self.I_xx = (1.0 / 12.0) * self.mass * (0.15 * 0.15 + 0.15 * 0.15)  # (1.0 / 12.0) * mass * (height * height + depth * depth)
        self.I_yy = self.I_xx      # Inertia around y-axis
        self.I_zz = 2*self.I_xx      # Inertia around z-axis

        # The control parameters for the dynamics 
        self.roll_tau = 0.253
        self.pitch_tau = 0.267
        self.roll_gain = 6.101
        self.pitch_gain = 6.497
        self.yaw_tau = 0.425
        self.K_yaw = 1.8

        self.horizon = []  # only x,y,z
        self.w0, self.lbw, self.ubw = [], [], []
        self.lbg, self.ubg = [], []
        # Objective function and constraints
        self.Q = DM.eye(self.n_states)  # State cost matrix
        self.R = DM.eye(self.n_controls)  # Control cost matrix
        self.k = 0.76  # Lift constant  # The more it goes up the lower the thrust

        self.speed = initial_speed  #0.5 #35 #0.2  # 0.1  # (1 m/s = 3.6 km/h)
        self.speed_rviz = self.speed
        self.Duration = 25/self.speed # 75 
        self.Duration_rviz = 25/self.speed_rviz # for hexagon # 75 #20   15/0.2 = 75
        self.end_point = [0.0, 0.0, 1.0]
        self.initial_hover_position = None  # keeping the initial hover position

    
        q_x, q_y, q_z = 1.0, 1.0, 1.0 # 1.0, 1.0, 4.0 # Increased positional weights  
        q_vx, q_vy, q_vz = 0.0, 0.0, 0.0  # Kept the same for velocity
        q_phi, q_theta, q_psi = 0.8, 0.8, 0.8   # Orientation may not be as critical
        q_r = 0.0   # Angular rates are kept the same
   
        self.current_time = 0.0
        self.current_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # ovde da bi moglo da leti od ta

        # Add new weight matrices for MV targets and MV rate of change
        #self.R = diag(DM([0.6, 1.2, 1.2, 0.6]))  # Tuning weight for MV target # thrust, roll, pitch, yaw_rate
        #self.R = diag(DM([1.0, 1.5, 1.5, 1.0])) 
        #self.R = diag(DM([0.2, 0.6, 0.6, 0.2])) 
        #self.R = diag(DM([0.1, 0.5, 0.5, 0.1])) 
        
        # LINKOPING VICON SYSTEM - WORKED TUNNING ON MAT2
        #self.R = diag(DM([0.2, 0.5, 0.5, 0.1])) 

        # working on tuning in granso
        #self.R = diag(DM([0.1, 0.5, 0.5, 0.1])) 
        #self.R = diag(DM([0.3, 0.5, 0.5, 0.1])) 
        #self.R = diag(DM([0.3, 0.6, 0.6, 0.2])) 

        self.R = diag(DM([0.6, 0.6, 0.6, 0.6])) 

        self.have_position = False
        self.have_target_position =False
        self.init_speed_flag = False


        if trajectory_type == "spline":
            # Define the 5 points in XYZ space for the spline
            self.num_points=5
            self.k = 0.75  # Lift constant  # The more it goes up the lower the thrust
            #q_x, q_y, q_z = 0.5, 0.9, 0.9 # 1.0, 1.0, 4.0 # Increased positional weights
            q_x, q_y, q_z = 1.0, 1.0, 10.0
            q_vx, q_vy, q_vz = 2.5, 2.5, 2.5 
            #start_point = np.array(self.start_point).reshape(1, -1)  # Ensure start_point is a 2D array with shape (1, 3)
            #print("Start point START POINT START POINT START POINT OF SPLINE", start_point)
   
            start_point = [0.0, 0.0, 2.0]
              # Define the other points
            other_points  = np.array([
                [2.0, 2.0, 2.0],  # Example point
                [4.0, -2.0, 4.0], # Example point
                [6.0, 3.0, 3.0],  # Example point
                [8.0, 0.0, 2.0]   # End point
            ]) 

            # Combine start_point with other_points
            self.points = np.vstack([start_point, other_points])    
            if use_nmpc==True:
                q_x, q_y, q_z = 1.0, 2.0, 2.0 
        
        if trajectory_type == "long_spline":
            self.k = 0.75  # Lift constant  # The more it goes up the lower the thrust
            q_x, q_y, q_z = 1.0, 1.0, 10.0
            q_vx, q_vy, q_vz = 0.5, 0.5, 8.5 
            
            self.num_points=9
            # Define the 5 points in XYZ space for the spline
            self.points = np.array([
                #[0.0, 0.0, 2.0],  # Start point
               #[2.0, 2.0, 2.0],  # Example point
               # [4.0, -2.0, 3.0], # Example point
               # [6.0, 3.0, 1.0],  # Example point
               # [8.0, 1.0, 4.0], # Example point
                #[10.0, 3.0, 3.0],  # Example point     
               #[12.0, 4.0, 1.0],  # Example point
                #[14.0, -1.0, 3.0], # Example point
               # [16.0, -3.0, 7.4],  # Example point  

                [18.0, 5.0, 8.0],  # Example point  
                [25.0, 6.0, 8.4],  # Start point
                [27.0, 8.0, 10.0],  # Example point
                [29.0, 4.0, 11.0], # Example point
                [31.0, 9.0, 9.0],  # Example point
                [33.0, 7.0, 12.0], # Example point
                [35.0, 9.0, 11.0],  # Example point     
                [37.0, 10.0, 9.0],  # Example point
                [39.0, 5.0, 11.0], # Example point
            ])
            if use_nmpc==True:
                  q_x, q_y, q_z = 1.0, 2.0, 2.0 

        if trajectory_type == "circle_xy":
            self.radius = 1.2
            self.k = 0.76  # Lift constant  # The more it goes up the lower the thrust
            q_x, q_y, q_z = 1.0, 2.0, 2.0 # 1.0, 1.0, 4.0 # Increased positional weights    
            if use_nmpc==True:
                q_x, q_y, q_z = 2.0, 2.0, 1.0 # 1.0, 1.0, 4.0 # Increased positional weights
                q_vx, q_vy, q_vz =  0.6, 0.6, 0.6 # Kept the same for velocity
                q_phi, q_theta, q_psi = 0.8, 0.8, 1.6   # Orientation 

        if trajectory_type == "sinusoidal":
            self.k = 0.75  # Lift constant  # The more it goes up the lower the thrust
            q_x, q_y, q_z = 1.0, 2.0, 10.0 # 1.0, 1.0, 4.0 # Increased positional weights    
            if use_nmpc==True:
                q_x, q_y, q_z = 2.0, 3.2, 2.0 

        if trajectory_type == "rectangle":
            self.length_x = 2.5
            self.length_y = 2.5
            self.k = 0.76  # Lift constant  # The more it goes up the lower the thrust
            q_x, q_y, q_z = 1.0, 1.0, 10.0 # 1.0, 1.0, 4.0 # Increased positional weights  
            if use_nmpc==True:
                q_x, q_y, q_z = 1.0, 1.0, 1.0 

        if trajectory_type == "hexagon":
            self.side_length = 1.2
            self.k = 0.76  # Lift constant  # The more it goes up the lower the thrust
            q_x, q_y, q_z = 1.0, 1.0, 10.0 # 1.0, 1.0, 4.0 # Increased positional weights    

            if use_nmpc==True:
                q_x, q_y, q_z = 1.0, 1.0, 1.0 

        if trajectory_type == "line_xy":
            self.length_leg = 5
            self.length_base = 5
            self.Duration = (self.length_leg + self.length_base)/self.speed  #length_leg+length_base
            self.Duration_rviz = self.Duration
            self.speed_rviz = self.speed
            q_x, q_y, q_z = 1.0, 1.0, 10.0
            if use_nmpc==True:
                q_x, q_y, q_z = 1.0, 1.0, 4.0 

        if trajectory_type == "step_z":
            self.step_height = 1.5
            self.Duration = self.step_height/self.speed  # step_height
            self.Duration_rviz = self.Duration
            self.speed_rviz = self.speed
            if use_nmpc==True:
                q_x, q_y, q_z = 1.0, 1.0, 4.0 

        if trajectory_type == "step_xyz":
            self.step_height_x = 1.5
            self.step_height_y = 1.5
            self.step_height_z = 1.5
            self.Duration = (abs(self.step_height_x) + abs(self.step_height_y) + abs(self.step_height_z))/self.speed  # abs(step_height_x) + abs(step_height_y) + abs(step_height_z)
            self.Duration_rviz = self.Duration
            self.speed_rviz = self.speed
            if use_nmpc==True:
                q_x, q_y, q_z = 1.0, 1.0, 4.0 

        if trajectory_type == "go_to_point":
            self.duration_const = 1.0    
            q_x, q_y, q_z = 1.0, 1.0, 10.0
            q_vx, q_vy, q_vz = 2.5, 2.5, 2.5 
            if use_nmpc==True:
                q_x, q_y, q_z = 1.0, 1.0, 4.0 

        if trajectory_type == "hover":   
            self.Duration = self.dt
            q_x, q_y, q_z = 1.0, 1.0, 10.0
            if use_nmpc==True:
                q_x, q_y, q_z = 1.0, 1.0, 1.0 
        self.speed_simulation = self.speed
        self.Duration_simulation = self.Duration

        # Objective function and constraints
        self.Q = diag(DM([q_x, q_y, q_z, q_phi, q_theta, q_psi, q_vx, q_vy, q_vz, q_r]))
         
        print("The speed is:", self.speed)
        print("The Duration is", self.Duration)
        print("The actual values of q_x, q_y, q_z",q_x, q_y, q_z)

        #print("Matrix Q:\n", Q)
        #print("Matrix R:\n", R)
        #print("Matrix Q_final:\n", Q_final)
        
        # Control input limits
        self.max_thrust = 60
        self.max_roll = np.math.radians(20.0)
        self.max_pitch = np.math.radians(20.0)
        self.min_roll = -np.math.radians(20.0)
        self.min_pitch = -np.math.radians(20.0)
        self.max_yaw = 5.0/6.0*np.pi #np.inf      # According to doc
        self.min_thrust = 30 #-np.inf                    # 37 it hovers...0-80
        self.min_yaw = -5.0/6.0*np.pi #-np.inf
        #self.u_min = [min_thrust, min_roll, min_pitch, min_yaw]
        #self.u_max = [max_thrust, max_roll, max_pitch, max_yaw]

        # CLOSED LOOP SIMULATION
        self.initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.u_initial_state = [initial_thrust, 0.0, 0.0, 0.0]
        
        print("HELP IS THIS GETTING TRIGGERED EVEN? WHAT IS HAPPENING?")
        print("self.u_inital_state", self.u_initial_state)
        self.x_new_w0 = np.zeros((self.n_states*(self.N))).tolist()
        self.u_new = np.zeros((self.n_controls*(self.N))).tolist()
        self.u_new[:0] += self.u_initial_state
        self.x_new_w0[:0] += self.initial_state

        # OBSTACLES - SPHERE            # 0.0 0.0 2.0
        self.obstacle_center = np.array([4.0, -2.0, 2.8])  # Obstacle position
        self.obstacle_radius = 0.75  # Obstacle radius
        
        # OBSTACLES - SPHERE
        # Define multiple obstacles
        self.obstacles = [
            {'center': np.array([27.0, 8.0, 9.9]), 'radius': 0.5},
            {'center': np.array([29.0, 4.0, 11.0]), 'radius': 0.6},
            {'center': np.array([37.5, 10.0, 9.0]), 'radius': 1.2},
            # Add more obstacles as needed
        ]
        self.safety_distance = 0.5 #1.25  # Safety distance from the obstacle


    def init_speed(self):

        if self.trajectory_type == "go_to_point":
            print("===============CURRENT POS:", self.current_state)
            print("===============END POS:", self.end_point)        
            x1 = np.array(self.current_state[:3])
            x2 = np.array(self.end_point)
            euclidean_distance = np.linalg.norm(x2 - x1)
            print("DIST:", euclidean_distance)
            self.Duration = euclidean_distance/self.speed  
            self.Duration_rviz = self.Duration

        if self.trajectory_type == "spline":

            time_points = np.linspace(0, 10, self.num_points)
            # Create B-splines for each dimension
            spline_x = make_interp_spline(time_points, self.points[:, 0], k=3)
            spline_y = make_interp_spline(time_points, self.points[:, 1], k=3)
            spline_z = make_interp_spline(time_points, self.points[:, 2], k=3)

            tp = np.linspace(0, 10, 1000)
            l = 0.0
            for i in range(1, len(tp)):
                t0 = tp[i-1]
                t1 = tp[i]
                dx = spline_x(t1) - spline_x(t0)
                dy = spline_y(t1) - spline_y(t0)
                dz = spline_z(t1) - spline_z(t0)
                l += math.sqrt(dx*dx + dy*dy+ dz*dz)
                #print(t0, t1, dx, dy, dz, l)

            self.Duration = l/self.speed
            self.Duration_rviz = self.Duration

        if self.trajectory_type == "long_spline":

            time_points = np.linspace(0, 10, self.num_points)
            # Create B-splines for each dimension
            spline_x = make_interp_spline(time_points, self.points[:, 0], k=3)
            spline_y = make_interp_spline(time_points, self.points[:, 1], k=3)
            spline_z = make_interp_spline(time_points, self.points[:, 2], k=3)

            tp = np.linspace(0, 10, 1000)
            l = 0.0
            for i in range(1, len(tp)):
                t0 = tp[i-1]
                t1 = tp[i]
                dx = spline_x(t1) - spline_x(t0)
                dy = spline_y(t1) - spline_y(t0)
                dz = spline_z(t1) - spline_z(t0)
                l += math.sqrt(dx*dx + dy*dy+ dz*dz)
                #print(t0, t1, dx, dy, dz, l)

            self.Duration = l/self.speed
            self.Duration_rviz = self.Duration

        if self.trajectory_type == "hexagon":
            self.Duration = (6*self.side_length + self.side_length)/self.speed  #  6×side_length
            self.Duration_rviz = self.Duration


        if self.trajectory_type == "circle_xy":
            self.Duration = 2*np.pi*self.radius/self.speed  # 2pi×radius
            self.Duration_rviz = self.Duration

        if self.trajectory_type == "sinusoidal": # or self.trajectory_type == "old_sinusoidal":
            # Define the time points for the sinusoidal trajectory
            num_points = 1000
            time_points = np.linspace(0, 10, num_points)
            
            # Define the sinusoidal path
            x_points = 2.5 * np.sin(2 * np.pi * time_points / 10)
            y_points = 2.5 * np.sin(2 * np.pi * time_points / 10) * np.cos(2 * np.pi * time_points / 10)
            z_points = 1 * np.cos(2 * np.pi * time_points / 10) + 2

            l = 0.0
            for i in range(1, len(time_points)):
                dx = x_points[i] - x_points[i - 1]
                dy = y_points[i] - y_points[i - 1]
                dz = z_points[i] - z_points[i - 1]
                l += math.sqrt(dx*dx + dy*dy + dz*dz)

            self.Duration = l/self.speed
            self.Duration_rviz = self.Duration


        if self.trajectory_type == "rectangle":
            self.Duration = (2*(self.length_x+self.length_y) + (self.length_x+self.length_y))/self.speed  # 2×(length_x+length_y)
            self.Duration_rviz = self.Duration

        self.init_speed_flag = True
    
    def notify_obstacle_center(self, r_x, r_y, r_z):
        self.obstacles[0]['center'] = [r_x, r_y, r_z]

    def notify_fly_to_point(self, x, y, z):
        self.end_point[0] = x
        self.end_point[1] = y
        self.end_point[2] = z
        self.have_target_position = True
            
    def reset(self):
        self.current_time = 0.0
        self.u_last = self.mv_target
        self.horizon = []

    def notify_velocity(self, vx, vy, vz):
        #print("VELOCITY:", vx, vy, vz)
        self.current_state[6] = vx
        self.current_state[7] = vy
        self.current_state[8] = vz
    
    def notify_position(self, x, y, z):
        #print("POSITION:", x, y, z)
        self.current_state[0] = x
        self.current_state[1] = y
        self.current_state[2] = z
        if not self.have_position:
            self.have_position = True
            self.start_point = self.current_state[:3]
            colorama.init(autoreset=True) 
            print(Fore.GREEN + "LOOOK HEREE the start_point:", self.start_point)

    def notify_angles(self, roll, pitch, yaw):
        #print(f"ANGLES: {math.degrees(roll), math.degrees(pitch), math.degrees(yaw)}")
        self.current_state[3] = roll
        self.current_state[4] = pitch
        self.current_state[5] = yaw
        
    def notify_attitude(self, x, y, z, w):
        pass

    def notify_angle_rates(self, x, y, z):
        self.current_state[9] = z

    def notify_yaw_rate(self, value):
        self.current_state[9] = value

    def drone_dynamics(self, x1, u):
        # Extract states   - 10 STATES x,y,z, roll, pitch, yaw, xdot, ydot, zdot, yaw_rate
        pos = x1[0:3]  # Position: x, y, z
        angles = x1[3:6]  # Orientation: phi, theta, psi
        vel = x1[6:9]  # Linear velocities: xdot, ydot, zdot
        ang_vel = x1[9]  # Angular velocities: psidot - yaw_rate

        # Control inputs
        thrust = u[0]  # Total thrust
        roll_ref = u[1]  # Roll angle in radians
        pitch_ref = u[2]  # Pitch angle in radians
        yaw_rate_ref = u[3]  # Yaw rate in radians per second

        # Rotation matrix from body frame to inertial frame
        R = self.rotation_matrix(angles)

        # Translational dynamics
        # Drag coefficients (example values, adjust as needed)
        b_x = 0.1  # Drag coefficient in x direction
        b_y = 0.1  # Drag coefficient in y direction
        b_z = 0.01  # Drag coefficient in z direction

        # Drag acceleration in body frame
        drag_acc_intertial = -vertcat(b_x * vel[0], b_y * vel[1], b_z * vel[2])
        # Gravity vector in inertial frame
        gravity_vector = vertcat(0, 0, -self.gravity)
        # Thrust vector in body frame
        thrust_vector = vertcat(0, 0, self.k*thrust)
        
        # Force in COM frame
        translational_dynamics = gravity_vector + mtimes(R, thrust_vector) / self.mass + mtimes(R, drag_acc_intertial)

        # Rotational dynamics
        # Assuming simple proportional control for roll and pitch, and direct yaw rate control
        droll = (1 / self.roll_tau) * (self.roll_gain*(roll_ref - angles[0]))
        dpitch = (1 / self.pitch_tau) * (self.pitch_gain*(pitch_ref - angles[1]))
        dyaw = ang_vel 
        
        ddyaw = (1 / self.yaw_tau) * (self.K_yaw * (yaw_rate_ref - ang_vel)) 

        rotational_dynamics = vertcat(droll, dpitch, dyaw)
        
        f = vertcat(vel, rotational_dynamics, translational_dynamics, ddyaw)

        return f
   
    # Rotation matrix function
    def rotation_matrix(self, angles):
        phi, theta, psi = angles[0], angles[1], angles[2]
        # Rotation matrices for Rz, Ry, Rx
        Rz = vertcat(horzcat(cos(psi), -sin(psi), 0),
                        horzcat(sin(psi), cos(psi), 0),
                        horzcat(0, 0, 1))
        Ry = vertcat(horzcat(cos(theta), 0, sin(theta)),
                        horzcat(0, 1, 0),
                        horzcat(-sin(theta), 0, cos(theta)))
        Rx = vertcat(horzcat(1, 0, 0),
                        horzcat(0, cos(phi), -sin(phi)),
                        horzcat(0, sin(phi), cos(phi)))
        R_dyn = mtimes(Rz, mtimes(Ry, Rx))
        return R_dyn


    '''
    # TRAJECTORY INFORMATION HERE
    '''   
    # Define a function to generate the reference trajectory (similar to MATLAB)
    def QuadrotorReferenceTrajectory(self, t, trajectory_type, Duration_ref, speed_ref):  # USE THE GLOABL ONE FIX
        if trajectory_type == 'go_to_point':
            # Define start and end points
            start_point = np.array(self.start_point)
            end_point = np.array(self.end_point)

            #euclidean_distance = np.linalg.norm(end_point - start_point)
            #baseline_duration = euclidean_distance / speed_ref  
            #Duration_ref = baseline_duration * self.duration_const
            # Define the points for the spline in each dimension
            points_x = np.array([start_point[0], end_point[0]])
            points_y = np.array([start_point[1], end_point[1]])
            points_z = np.array([start_point[2], end_point[2]])

            # Time points for the beginning and end of the trajectory
            #print("*****************:", Duration_ref)
            time_points = np.linspace(0, Duration_ref, len(points_x))
            # time_point = [0, Duration_ref]

            # Create B-splines for each dimension
            spline_x = make_interp_spline(time_points, points_x, k=1)  # Cubic B-spline --- from the paper you saw
            spline_y = make_interp_spline(time_points, points_y, k=1)  
            spline_z = make_interp_spline(time_points, points_z, k=1)  

            # Compute the position at time t using the spline
            x = spline_x(np.clip(t, 0, Duration_ref))
            y = spline_y(np.clip(t, 0, Duration_ref))
            z = spline_z(np.clip(t, 0, Duration_ref))

            xdot = spline_x.derivative()(np.clip(t, 0, Duration_ref))
            ydot = spline_y.derivative()(np.clip(t, 0, Duration_ref))
            zdot = spline_z.derivative()(np.clip(t, 0, Duration_ref))
        elif trajectory_type == 'hover':
            if self.initial_hover_position is None:
                self.initial_hover_position = self.current_state[:3]  
            
            # Use the initial hover position for the entire duration
            x = np.full_like(t, self.initial_hover_position[0])  
            y = np.full_like(t, self.initial_hover_position[1])  
            z = np.full_like(t, self.initial_hover_position[2])  

        elif trajectory_type == 'spline':


            # Time points for the spline (equally spaced)
            time_points = np.linspace(0, Duration_ref, self.num_points)
            # Create B-splines for each dimension
            spline_x = make_interp_spline(time_points, self.points[:, 0], k=3)
            spline_y = make_interp_spline(time_points, self.points[:, 1], k=3)
            spline_z = make_interp_spline(time_points, self.points[:, 2], k=3)

            # Compute the position at time t using the spline
            x = spline_x(np.clip(t, 0, Duration_ref))
            y = spline_y(np.clip(t, 0, Duration_ref))
            z = spline_z(np.clip(t, 0, Duration_ref))

            xdot = spline_x.derivative()(np.clip(t, 0, Duration_ref))
            ydot = spline_y.derivative()(np.clip(t, 0, Duration_ref))
            zdot = spline_z.derivative()(np.clip(t, 0, Duration_ref))

        elif trajectory_type == 'long_spline':

            # Time points for the spline (equally spaced)
            time_points = np.linspace(0, Duration_ref, self.num_points)
            # Create a spline interpolation for each dimension
            spline_x = make_interp_spline(time_points, self.points[:, 0], k=3)
            spline_y = make_interp_spline(time_points, self.points[:, 1], k=3)
            spline_z = make_interp_spline(time_points, self.points[:, 2], k=3)
            # Compute the position at time t using the spline
            x = spline_x(np.clip(t, 0, Duration_ref))
            y = spline_y(np.clip(t, 0, Duration_ref))
            z = spline_z(np.clip(t, 0, Duration_ref))

            xdot = spline_x.derivative()(np.clip(t, 0, Duration_ref))
            ydot = spline_y.derivative()(np.clip(t, 0, Duration_ref))
            zdot = spline_z.derivative()(np.clip(t, 0, Duration_ref))

        elif trajectory_type == 'step_xyz':
            # Step trajectory in x, y, and z directions
            step_time_x = 5
            step_height_x = 1
            step_time_y = 5
            step_height_y = 1
            step_time_z = 5
            step_height_z = 1
            x = np.where(t >= step_time_x, step_height_x, 0)
            y = np.where(t >= step_time_y, step_height_y, 0)
            z = np.where(t >= step_time_z, step_height_z, 0)

        elif trajectory_type == 'spline':
            num_points=5
            # Define the 5 points in XYZ space for the spline
            points = np.array([
                [0.0, 0.0, 1.0],  # Start point
                [2.0, 2.0, 2.0],  # Example point
                [4.0, -2.0, 3.0], # Example point
                [6.0, 3.0, 2.0],  # Example point
                [8.0, 0.0, 1.0]   # End point
            ])
            # Time points for the spline (equally spaced)
            time_points = np.linspace(0, self.Duration, num_points)
            # Create a spline interpolation for each dimension
            spline_x = interp1d(time_points, points[:, 0], kind='cubic')
            spline_y = interp1d(time_points, points[:, 1], kind='cubic')
            spline_z = interp1d(time_points, points[:, 2], kind='cubic')
            # Compute the position at time t using the spline
            x = spline_x(np.clip(t, 0, self.Duration))
            y = spline_y(np.clip(t, 0, self.Duration))
            z = spline_z(np.clip(t, 0, self.Duration))

        elif trajectory_type == 'sinusoidal':
            # Define multiple points for a full cycle of sinusoidal motion
            num_points = 100
            time_points = np.linspace(0, Duration_ref, num_points)
            x_points = 2.5 * np.sin(2 * np.pi * time_points / Duration_ref)
            y_points = 2.5 * np.sin(2 * np.pi * time_points / Duration_ref) * np.cos(2 * np.pi * time_points / Duration_ref)
            z_points = 1 * np.cos(2 * np.pi * time_points / Duration_ref) + 2  # Keeping z variation
            
            spline_x = CubicSpline(time_points, x_points, bc_type='natural')
            spline_y = CubicSpline(time_points, y_points, bc_type='natural')
            spline_z = CubicSpline(time_points, z_points, bc_type='natural')
            x = spline_x(np.clip(t, 0, Duration_ref))
            y = spline_y(np.clip(t, 0, Duration_ref))
            z = spline_z(np.clip(t, 0, Duration_ref))

        elif trajectory_type == 'rectangle':
            # Define rectangle dimensions
            length_x = self.length_x
            length_y = self.length_y

            # Define corner points of the rectangle (closed loop)
            points = np.array([
                [0, 0, 3],  # Bottom-left
                [length_x, 0, 3],  # Bottom-right
                [length_x, length_y, 3],  # Top-right
                [0, length_y, 3],  # Top-left
                [0, 0, 3]  # Back to start to close the loop
            ])

            # Calculate the time to reach each corner
            duzina = 2 * (length_x + length_y)
            time_per_unit_length = Duration_ref / duzina
            times = np.cumsum([0] + [np.linalg.norm(points[i] - points[i - 1]) * time_per_unit_length for i in range(1, len(points))])

            # B-Spline interpolation for each dimension
            spline_x = make_interp_spline(times, points[:, 0], k=1) 
            spline_y = make_interp_spline(times, points[:, 1], k=1)
            spline_z = make_interp_spline(times, points[:, 2], k=1)
            # Evaluate spline at time t
            max_time = times[-1]
            t_mod = np.mod(t, max_time)  # Wrap time for cyclic behavior
            x = spline_x(t_mod)
            y = spline_y(t_mod)
            z = spline_z(t_mod)

        elif trajectory_type == 'hexagon':
            # Hexagon parameters
            num_sides = 6
            angle = 2 * np.pi / num_sides  # Each angle in radians
            side_length = self.side_length

            # Define corner points of the hexagon
            points = [np.array([side_length * np.cos(i * angle), side_length * np.sin(i * angle), 2]) for i in range(num_sides)]
            points.append(points[0])  # Append the first point at the end to close the loop

            # Calculate the time to reach each corner
            total_perimeter = num_sides * side_length
            time_per_unit_length = Duration_ref / total_perimeter
            times = np.cumsum([0] + [side_length * time_per_unit_length for _ in range(num_sides)])  # Time for each segment

            # B-Spline interpolation for each dimension
            spline_x = make_interp_spline(times, [p[0] for p in points], k=1)  # Linear B-spline
            spline_y = make_interp_spline(times, [p[1] for p in points], k=1)
            spline_z = make_interp_spline(times, [p[2] for p in points], k=1)

            # Evaluate spline at time t
            max_time = times[-1]
            t_mod = np.mod(t, max_time)  # Wrap time for cyclic behavior
            x = spline_x(t_mod)
            y = spline_y(t_mod)
            z = spline_z(t_mod)

        elif trajectory_type == 'line_xy':
            # Points to define the L shape: start, corner, and end points
            points_x = [0, 0, self.length_base]
            points_y = [0, self.length_leg, self.length_leg]
            points_z = [3, 3, 3]  # Constant height at 3 

            # Time distribution should align with the proportions of leg and base
            time_points = [0, Duration_ref * self.length_leg / (self.length_leg + self.length_base), Duration_ref]

            # Create cubic splines for each dimension
            spline_x = CubicSpline(time_points, points_x, bc_type='natural')
            spline_y = CubicSpline(time_points, points_y, bc_type='natural')
            spline_z = CubicSpline(time_points, points_z, bc_type='natural')
            # Compute the position at time t using the spline
            #x = spline_x(np.clip(t, 0, Duration_ref))
            #y = spline_y(np.clip(t, 0, Duration_ref))
            #z = spline_z(np.clip(t, 0, Duration_ref))

            x = spline_x(np.minimum(t, Duration_ref))
            y = spline_y(np.minimum(t, Duration_ref))
            z = spline_z(np.minimum(t, Duration_ref))

        elif trajectory_type == 'circle_xy':
            omega = speed_ref
            # Define the center of the circle
            x_center = 0.0  # Example x-coordinate offset
            y_center = 0.0  # Example y-coordinate offset
            z_center = 2.0  # Constant z height, adjusted if needed

            num_points = 100
            # Create angles for points around the circle
            angles = np.linspace(0, 2 * np.pi, num_points + 1)
            # Circle points with offset
            points_x = self.radius * np.cos(angles) + x_center
            points_y = self.radius * np.sin(angles) + y_center
            points_z = np.full_like(points_x, z_center)  # Constant z height, e.g., z=2

            # Time points should cover the duration of one full circle
            time_points = np.linspace(0, Duration_ref, num_points + 1)

            # Create cubic splines for each dimension
            spline_x = CubicSpline(time_points, points_x, bc_type='periodic')
            spline_y = CubicSpline(time_points, points_y, bc_type='periodic')
            spline_z = CubicSpline(time_points, points_z, bc_type='periodic')
            # Compute the position at time t using the spline
            x = spline_x(np.clip(t, 0, Duration_ref))
            y = spline_y(np.clip(t, 0, Duration_ref))
            z = spline_z(np.clip(t, 0, Duration_ref))

        elif trajectory_type == 'step_z':
            # Step trajectory
            step_time = 5
            x = np.zeros_like(t)
            y = np.zeros_like(t)
            z = np.where(t >= step_time, self.step_height, 0)
            
        elif trajectory_type == 'step_xyz':
            # Step trajectory in x, y, and z directions
            step_time_x = 5
            step_time_y = 5
            step_time_z = 5
            x = np.where(t >= step_time_x, self.step_height_x, 0)
            y = np.where(t >= step_time_y, self.step_height_y, 0)
            z = np.where(t >= step_time_z, self.step_height_z, 0)

        else:
            # Default to hover mode if unknown type
            if self.initial_hover_position is None:
                self.initial_hover_position = self.current_state[:3]  
            
            # Use the initial hover position for the entire duration
            x = np.full_like(t, self.initial_hover_position[0])  
            y = np.full_like(t, self.initial_hover_position[1])  
            z = np.full_like(t, self.initial_hover_position[2])  
            xdot = np.zeros_like(t)
            ydot = np.zeros_like(t)
            zdot = np.zeros_like(t)
        # Initialize other states to zero
        phi = np.zeros_like(t)
        theta = np.zeros_like(t)
        psi = np.zeros_like(t)
        #xdot = np.zeros_like(t)
        #ydot = np.zeros_like(t)
        #zdot = np.zeros_like(t)
        psidot = np.zeros_like(t)

        # Combine all states into a single array
        xdesired = np.vstack([x, y, z, phi, theta, psi, xdot, ydot, zdot, psidot])

        return xdesired

    def QuadrotorReferenceTrajectoryMatrix(self, current_time, trajectory_type, Duration_ref, speed_ref):
        # Generate N+1 timesteps from current_time to current_time + N*dt
        timesteps = np.linspace(current_time, current_time + self.N*self.dt, self.N)
        trajectory_matrix = np.zeros((self.n_states, self.N))

        for i, t in enumerate(timesteps):
            trajectory_point = self.QuadrotorReferenceTrajectory(t, trajectory_type, Duration_ref, speed_ref)
            trajectory_matrix[:, i] = trajectory_point.flatten()

        return trajectory_matrix

    def get_trajectory(self, current_time, trajectory_type, Duration_ref, speed_ref):

        x_desired_matrix = self.QuadrotorReferenceTrajectoryMatrix(current_time, trajectory_type, Duration_ref, speed_ref)
        x_desired_matrix = x_desired_matrix.transpose()
        # Flatten the matrix to pass it to the solver if necessary
        reftraj = x_desired_matrix.flatten().tolist()

        return reftraj
    
    def calculate_objective(self, x_param, u_param, x_desired_param):
        # Calculate state error using the parameterized x_desired
        state_error_weighted = mtimes(self.Q, (x_param - x_desired_param))
        control_effort_weighted = mtimes(self.R, u_param)
        # Objective function with weighted components
        L = dot(state_error_weighted, state_error_weighted) + dot(control_effort_weighted, control_effort_weighted)
        
        # Add obstacle avoidance penalty
        for obstacle in self.obstacles:
            distance_to_obstacle = sqrt((x_param[0] - obstacle['center'][0])**2 + 
                                        (x_param[1] - obstacle['center'][1])**2 + 
                                        (x_param[2] - obstacle['center'][2])**2)
            eta = 1500.0
            repulsive_cost = if_else(distance_to_obstacle < (obstacle['radius'] + self.safety_distance),
                                    0.5 * eta * ((1 / distance_to_obstacle) - (1 / (obstacle['radius'] + self.safety_distance)))**2,
                                    0)
            L += repulsive_cost
        return L

    def create_nmpc_problem(self, use_rk4=False, explicit_euler=True):
        # Initialize NLP variables
        w, p = [], []
        J = 0  # Objective function
        g= []
        #print("BAG 0")
        # "Lift" initial conditions
        #Xk = MX.sym('X0', self.n_states*(self.N+1))
        #Xk = Xk[0:self.n_states]       
        Xk = MX.sym('X0', self.n_states)
        w.append(Xk)
        #print("BAG 1")    
        # Set the lower and upper bounds to the exact initial state values
        self.lbw += self.initial_state #[-inf]*self.n_states
        self.ubw += self.initial_state #[inf]*self.n_states
        self.w0 += self.initial_state #[0]*self.n_states

        #print("Xk", Xk)
        #print("Yref", Yref)
        #print("Uk", Uk)
        dynamics_func = Function('dynamics', [self.x1, self.u], [self.drone_dynamics(self.x1, self.u)])
        objective_func = Function('objective', [self.x1, self.u, self.x_desired_param], [self.calculate_objective(self.x1, self.u, self.x_desired_param)])
        
        # Formulate the NLP
        for k in range(self.N):

            #print("BAG 3")
            Uk = MX.sym(f'Uk_{k}', self.n_controls)
            w.append(Uk)
            self.lbw += [self.min_thrust, self.min_roll, self.min_pitch, self.min_yaw]
            self.ubw += [self.max_thrust, self.max_roll, self.max_pitch, self.max_yaw]
            # Use self.u_new for the initial guess of controls
            self.w0 += [0]*self.n_controls
            Yref = MX.sym(f'Yref_{k}', self.n_states)
            p.append(Yref)
            #print("BAG 4")

            #print("Yref", Yref)
            #print("Uk", Uk)
            #print("Xk", Xk)
            #print("Uk size:", Uk.size())
            #print("Yref size:", Yref.size())
            #print("Xk size:", Xk.size())

            # Integrate dynamics
            if use_rk4:
                # Fixed step Runge-Kutta 4 integrator         # manually written runge kutta 
                M = 4 # RK4 steps per interval
                DT = self.T/self.N/M
                Q = 0
                X = Xk
                #print("BAG 5")
                for j in range(M):
                    k1, L1 = dynamics_func(X, Uk), objective_func(X, Uk, Yref)
                    k2, L2 = dynamics_func(X + DT/2 * k1, Uk), objective_func(X + DT/2 * k1, Uk, Yref)
                    k3, L3 = dynamics_func(X + DT/2 * k2, Uk), objective_func(X + DT/2 * k2, Uk, Yref)
                    k4, L4 = dynamics_func(X + DT * k3, Uk), objective_func(X + DT * k3, Uk, Yref)

                    X += DT/6 * (k1 + 2*k2 + 2*k3 + k4)
                    Q += DT/6 * (L1 + 2*L2 + 2*L3 + L4)
                #print("BAG 6")
                F = Function('F', [Xk, Uk, Yref], [X, Q], ['x0', 'u', 'yref'], ['xf', 'qf'])
                Fk = F(x0 = Xk, u = Uk, yref = Yref)
                Xk_end = Fk['xf']
                J += Fk['qf']  # Accumulate objective
            elif explicit_euler:
                # Choose the Explicit Euler method for dynamics integration
                DT = self.T / self.N
                k1, L1 = dynamics_func(Xk, Uk), objective_func(Xk, Uk, Yref)

                # Explicit Euler step
                X = Xk + DT * k1
                Q = L1  # Objective function contribution for this step

                F = Function('F', [Xk, Uk, Yref], [X, Q], ['x0', 'u', 'yref'], ['xf', 'qf'])
                Fk = F(x0=Xk, u=Uk, yref=Yref)
                Xk_end = Fk['xf']
                J += Fk['qf']
            else:
                # Use the CVODES integrator for the non-RK4 path
                # Correct setup for calling the integrator
                dae = {'x': Xk, 'p': vertcat(Uk, Yref), 'ode': self.drone_dynamics(Xk, Uk), 'quad': self.calculate_objective(Xk, Uk, Yref)}
                opts = {'tf': self.dt}  # Define the time step for integration
                integrator_instance = integrator('F', 'cvodes', dae, opts)
                Fk = integrator_instance(x0=Xk, p=vertcat(Uk, Yref))
                Xk_end = Fk['xf']
                J += Fk['qf']

            # New NLP variable for state at the end of the interval if not the last interval
            if k < self.N - 1:
                Xk = MX.sym('X_' + str(k+1), self.n_states)
                w.append(Xk)
                self.lbw += [-inf]*self.n_states
                self.ubw += [inf]*self.n_states
                self.w0 += [0]*self.n_states
                #print("BAG 32")
                #print("Xk", Xk)
                #Yref = MX.sym('Yref_' + str(k+1), self.n_states)
                #p.append(Yref)
                #print("Yref", Yref)

            '''
            # Calculate the distance from the obstacle
            distance_to_obstacle = sqrt((Xk[0] - self.obstacle_center[0])**2 + 
                                        (Xk[1] - self.obstacle_center[1])**2 + 
                                        (Xk[2] - self.obstacle_center[2])**2)

            # ------------------------------------------------------------------
            # COMMENT OUT IF TOO MUCH - ARTIFICIAL POTENTIAL FIELDS
            # Add repulsive potential to the cost function
            #eta = 50.0 # This is a scaling factor that determines how strong the repulsive force is. 
            # Increasing eta makes the drone react more strongly to the obstacle
            #repulsive_cost = if_else(distance_to_obstacle < (self.obstacle_radius + self.safety_distance),
            #                        0.5 * eta * ((1 / distance_to_obstacle) - (1 / (self.obstacle_radius + self.safety_distance)))**2,

            #print("PAY ATTENTION")    
            #print("Distance to Obstacle: %f \n", distance_to_obstacle)
            #print("Repulsive Cost: %f \n", repulsive_cost)
            #                                   0)
            #J += repulsive_cost
            # ------------------------------------------------------------------
            # IF THIS SLACK THING DOESNT WOTK USE THE BELOW CODE
            # Add slack variable for obstacle avoidance
            slack = MX.sym(f'slack_{k}')
            w.append(slack)
            self.lbw.append(0)  # Slack variable should be non-negative
            self.ubw.append(np.inf)
            self.w0.append(0)

            # Obstacle avoidance constraint with slack
            g.append(distance_to_obstacle - (self.obstacle_radius + self.safety_distance) + slack)
            self.lbg.append(0)  # Ensure positive distance
            self.ubg.append(np.inf)

            # Penalize the slack variable in the objective function
            #J += 1e9 * slack  # Increase the penalty coefficient

            # Penalize the slack variable in the objective function
            #J += 1e5 * slack  # Adjust the penalty coefficient as needed - the bigger the slack
            # This means that in situations where the drone might otherwise violate the obstacle avoidance constraint, 
            # the slack variable can take a non-zero value to make the constraint feasible.
            #J += 1e2 * slack
            J += 1e0 * slack  # Reducing the penalty coefficient we get more Hz 
            #J += 1e0 * slack  # Further reducing the penalty coefficient
            #J += 1e-10 * slack  # Even further reducing the penalty coefficient - it will use the slack minimaly

            # ------------------------------------------------------------------

            # Add constraint for obstacle avoidance - UNCOMMENT AND USE INSTEAD OF ABOVE CODE ADDED FOR SLACK
            #g.append(distance_to_obstacle - (self.obstacle_radius + self.safety_distance))
            #self.lbg.append(0)  # Lower bound for the constraint, ensuring positive distance
            #self.ubg.append(np.inf)  # Upper bound, no maximum limit

            # ------------------------------------------------------------------
            '''            
            # Add equality constraint for dynamics continuity
            # Now append the continuity constraint correctly
            g.append(Xk_end - Xk)
            self.lbg += [0]*self.n_states
            self.ubg += [0]*self.n_states

        #print("OUT OF LOOOP")    
        #print("Size of w0:", len(self.w0))
        #print("Size of lbw:", len(self.lbw))
        #print("Size of ubw:", len(self.ubw))
        #print("Size of w:", len(w))
        #print("Size of g:", len(g))
        #print("Size of p:", len(p))

        # Define solver options
        # Define solver options
        opts = {
                'ipopt.print_level': 0, 'print_time': 1, #3 1
                'ipopt.warm_start_init_point': 'yes',  # Tell IPOPT to use warm start
                'ipopt.warm_start_bound_push': 1e-3,
                'ipopt.warm_start_mult_bound_push': 1e-3,
                'ipopt.mu_init': 1e-3,  # Initial value for the barrier parameter
                'ipopt.mu_strategy': 'adaptive',
                'ipopt.max_iter': 200,  # Maximum number of iterations
                'ipopt.tol': 1e-1, #2,       # Tolerance for convergence the bigger the number the tighter the tolerance  # RELAX THIS IF THE OBSTLE CRASHES
                'ipopt.acceptable_tol': 1e-1, #3, # Acceptable tolerance for convergence  # RELAX THIS IF THE OBSTLE CRASHES
                'ipopt.linear_solver': 'mumps', # 'mumps' #'ma27 reko bjorn' # Linear solver to be used}  # Increased print level
                'ipopt.nlp_scaling_method': 'gradient-based',
                'ipopt.nlp_scaling_max_gradient': 100.0,
                'ipopt.bound_push': 1e-3,    
                'ipopt.bound_frac': 1e-3,    
            }
        # Create an NLP solver
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(*p)}
        self.solver = nlpsol('solver', 'ipopt', prob, opts)


    def solve_nmpc(self, xk, yref, x_new_w0, u_new):

        t2 = time.time()

        #print("What is the current_state:", current_state)
        #self.w0[:self.n_states] = current_state

        # Update 'w0' with 'self.x_new_w0'
        for i in range(self.N):
            # Calculate the index in 'w0' where the current state prediction should be updated
            index = i * (self.n_states + self.n_controls)
            # Update 'w0' with the new state prediction
            self.w0[index:index+self.n_states] = self.x_new_w0[i*self.n_states:(i+1)*self.n_states]

        for i in range(self.N):
            # Calculate the starting index for the control inputs in 'w0' for each step
            control_index = self.n_states + i * (self.n_states + self.n_controls)
            # Update 'w0' with the new control inputs
            self.w0[control_index:control_index+self.n_controls] = self.u_new[i*self.n_controls:(i+1)*self.n_controls]

        self.lbw[:self.n_states] = xk
        self.ubw[:self.n_states] = xk

        yref_actual = np.concatenate([yref[start_index:start_index + self.n_states] for start_index in range(0, len(yref), self.n_states)]).tolist()
        #print("Type and size of yref_actual:", type(yref_actual), ", Length:", len(yref_actual))
         

        # Debugging information
        #print("Debug Information:")
        #print("Solver exists:", hasattr(self, 'solver'))
        #if hasattr(self, 'solver'):
        #    print("Solver type:", type(self.solver))
        #print("w0 length:", len(self.w0))
        #print("lbw length:", len(self.lbw))
        #print("ubw length:", len(self.ubw))
        #print("lbg length:", len(self.lbg))
        #print("ubg length:", len(self.ubg))
        #print("yref_actual length:", len(yref_actual))
        #print("GRESKA")

        # Solve the NMPC optimization problem
        sol = self.solver(x0=self.w0, p=yref_actual, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)

        t3 = time.time()
        print("SOLVER self.Duration:", t3-t2)
        
        if self.solver.stats()['success']:
            # Extract the full solution
            w_opt = sol['x'].full().flatten()
            return w_opt
        else:
            print("Solver failed:", self.solver.stats()['return_status'])
            return None

    def full_trajectory_rviz(self, trajectory_type='sinusoidal'):
        # Suppose you want to generate the full trajectory from t=0 to t=self.Duration
        num_points = int(self.Duration_rviz / self.dt)
        timesteps = np.linspace(0, self.Duration_rviz, num_points)
        # Initialize yref_full to store the full reference trajectory
        yref_full = np.zeros((3, num_points))  # we are only interested in x, y, z components

        # Generate the full reference trajectory for each time step
        for i, t in enumerate(timesteps):
            # Obtain the full state vector from the trajectory function
            x_desired = self.QuadrotorReferenceTrajectory(t, trajectory_type, self.Duration_rviz, self.speed_rviz)
            # Extract and store only the x, y, z components
            yref_full[:, i] = x_desired[:3, 0] 
        
        return yref_full, self.dt, self.Duration_rviz

    def tick(self, current_time, trajectory_type='sinusoidal'):
        print("Ajmo da VIDIMO U TICK current_state", self.current_state)

        yref = self.get_trajectory(current_time, trajectory_type, self.Duration, self.speed)
        #print("BEFORE WE GO TO THE SOLVER")
        #print("BEFORE WE GO TO THE SOLVER")
        # Setup NMPC problem once
        t0 = time.time()

        w_opt = self.solve_nmpc(self.current_state, yref, self.x_new_w0, self.u_new) 
        # Solve the NMPC optimization problem
        t1 = time.time()
        print("SOLVER_NMPC self.Duration:", t1-t0)
        
        if w_opt is None:
            print("Solver failed to find a solution.")
            # Handle the failure case, e.g., by returning early or using a fallback strategy
            return None, yref, self.current_state, None
        
        # Initialize arrays to store states and controls
        states = np.zeros((self.n_states, self.N))
        controls = np.zeros((self.n_controls, self.N))
        # Extract states and controls from w_opt
        for j in range(self.N):
            # States are at positions 0, n_controls+n_states, 2*(n_controls+n_states), ...
            states[:, j] = w_opt[j * (self.n_controls + self.n_states):(j * (self.n_controls + self.n_states)) + self.n_states]
            # Controls are immediately after each state
            #if j < N:  # No control after the last state
            controls[:, j] = w_opt[(j * (self.n_controls + self.n_states)) + self.n_states:(j + 1) * (self.n_controls + self.n_states)]
        #print(f"Time , Optimal Position full: {states}")
        #print(f"Time, Optimal Control full: {controls}")
        #print("Timestep:", current_time)
        # Extract and apply the first control action, simulate dynamics
        u_opt = controls[:, 0]

        prev_controls = controls[:, 1:]  # Remove the first control action
        last_control = controls[:, -1]  # Take the last control input from the shifted controls
        u_novo = np.hstack((prev_controls, last_control.reshape(-1, 1))) # Append the new control at the end
        u_novel = []
        # Loop through each column of u_novo
        for q in range(u_novo.shape[1]):  # Iterate over columns
            # Extract the element from each row of the current column and add to u_novel
            u_novel.extend(u_novo[:, q])
        self.u_new = np.array(u_novel).flatten().tolist()

        prev_states = states[:, 1:] # Remove the first state
        x_novo_w0 = np.hstack((np.array(self.current_state).reshape(-1, 1), prev_states))
        x_novel_w0 = []
        # Loop through each column of x_novo_w0
        for p in range(x_novo_w0.shape[1]):  # Iterate over columns
            # Extract the element from each row of the current column and add to u_novel
            x_novel_w0.extend(x_novo_w0[:, p])
        self.x_new_w0 = np.array(x_novel_w0).flatten().tolist()

        #print("AFTER THE SOLVER AND SIMULATED DYNAMICS")
        #print("The control input:", u_opt)
        self.current_time += self.dt

        return u_opt, yref, self.current_state, states

    def simulate_quadrotor_dynamics(self, x_current, u_opt, use_rk4=True):
        if use_rk4:
            # Define the RK4 integration steps using the current state and control input directly
            k1 = self.drone_dynamics(x_current, u_opt)
            k2 = self.drone_dynamics(x_current + self.dt/2.0 * k1, u_opt)
            k3 = self.drone_dynamics(x_current + self.dt/2.0 * k2, u_opt)
            k4 = self.drone_dynamics(x_current + self.dt * k3, u_opt)

            # Calculate the next state directly without creating a CasADi function
            x_next = x_current + (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            # Use CasADi's integrator
            dae = {'x': self.x, 'p': self.u, 'ode': self.drone_dynamics(self.x, self.u)}
            opts = {'tf': self.dt}  # Time step for the integrator
            integrator_instance = integrator('integrator', 'cvodes', dae, opts)
            x_next = integrator_instance(x0=x_current, p=u_opt)['xf']
        return DM(x_next).reshape((-1, 1))    



    def closed_loop_simulation(self, trajectory_type='go_to_point'):
        
        print(f"Running simulation with trajectory type: {trajectory_type}")

        x_current = self.initial_state
        current_time = 0.0            # Start time
        n_sim_steps = int(self.Duration_simulation/ self.dt)

        # Initialize histories
        xHistory = np.zeros((n_sim_steps+1, self.n_states))
        uHistory = np.zeros((n_sim_steps, self.n_controls))
        xDesiredHistory = np.zeros((n_sim_steps+1, self.n_states))  # Assuming same dimension as xHistory
        # Initial conditions
        xHistory[0, :] = self.initial_state

        for k in range(n_sim_steps):

            current_time = k * self.dt
            xHistory[k, :] = x_current
            # Update x_desired based on the current time
            x_zeljeno = self.get_trajectory(current_time, trajectory_type, self.Duration_simulation, self.speed_simulation)
            # Update the initial state in the optimization problem to the current state
            print("BEFORE WE GO TO THE SOLVER")

            # Update x_desired based on the current time - FOR PLOTTING
            x_desired_current = self.QuadrotorReferenceTrajectory(current_time, trajectory_type, self.Duration_simulation, self.speed_simulation)
            x_zelimo = DM(x_desired_current).reshape((-1, 1))  # Ensure correct shape/format
            x_desired = np.array(x_zelimo).flatten().tolist()
            xDesiredHistory[k, :] = x_desired

            #print("The actual trajectory:", x_current)
            #print("The size of x_current:", np.size(x_current))
            #print("The size of x_zeljeno:", np.size(x_zeljeno))
            #print("The size of self.u_new:", np.size(self.u_new))
            # Setup NMPC problem once
            w_opt = self.solve_nmpc(x_current, x_zeljeno, self.x_new_w0, self.u_new)
            # Solve the NMPC optimization problem
            #print("The solution w_opt:", w_opt)

            # Initialize arrays to store states and controls
            states = np.zeros((self.n_states, self.N))
            controls = np.zeros((self.n_controls, self.N))

            # Extract states and controls from w_opt
            for j in range(self.N):
                # States are at positions 0, n_controls+n_states, 2*(n_controls+n_states), ...
                states[:, j] = w_opt[j * (self.n_controls + self.n_states):(j * (self.n_controls + self.n_states)) + self.n_states]
                # Controls are immediately after each state
                #if j < N:  # No control after the last state
                controls[:, j] = w_opt[(j * (self.n_controls + self.n_states)) + self.n_states:(j + 1) * (self.n_controls + self.n_states)]

            # The last state is at the end
            #states[:, N] = w_opt[-n_states:]
            #print(f"Time , Optimal Position full: {states}")
            #print(f"Time, Optimal Control full: {controls}")
            print("Timestep:", current_time)

            # Extract and apply the first control action, simulate dynamics
            u_opt = controls[:, 0]
            u_optimalno = np.array(u_opt).flatten()
            uHistory[k, :] = u_optimalno
            #x_opt = states[:, 0]
            #print("The optimal states:", states)
            #print("The optimal controls:", controls)
            # Simulate the quadrotor dynamics for one time step using the first control action
            # This could be a call to a simulation function or dynamics_func with RK4 integration for dt_control
            x_next = self.simulate_quadrotor_dynamics(x_current, u_opt)
            # Correctly converting DM to list
            x_current = np.array(x_next).flatten().tolist()  # Convert DM to NumPy array and flatten to list
            # Store the simulated state
            xHistory[k+1, :] = x_current
            #print("The current state is x_current:", x_current)

            print("AFTER THE SOLVER AND SIMULATED DYNAMICS")
            #print("The optimal control is controls:", controls)
            print("The control input:", u_opt)
            print("The current state is x_current:", x_current)
            #print("The current desired trajectory:", x_desired)
            #print("The actual trajectory:", x_current)
            # After solving the NMPC problem and applying the first control action
            prev_controls = controls[:, 1:]  # Remove the first control action
            # Assuming controls are stored such that each column is a control vector for a time step
            last_control = controls[:, -1]  # Take the last control input from the shifted controls
            u_novo = np.hstack((prev_controls, last_control.reshape(-1, 1))) # Append the new control at the end
            u_novel = []
            # Loop through each column of u_novo
            for q in range(u_novo.shape[1]):  # Iterate over columns
                # Extract the element from each row of the current column and add to u_novel
                u_novel.extend(u_novo[:, q])
            self.u_new = np.array(u_novel).flatten().tolist()

            #print("X_CURRENT", x_current)
            #print("STATES", states)
            #print("the size of the states",np.size(states))
            prev_states = states[:, 1:] # Remove the first state
            x_novo_w0 = np.hstack((np.array(x_next).reshape(-1, 1), prev_states))
            x_novel_w0 = []
            # Loop through each column of x_novo_w0
            for p in range(x_novo_w0.shape[1]):  # Iterate over columns
                # Extract the element from each row of the current column and add to u_novel
                x_novel_w0.extend(x_novo_w0[:, p])
            self.x_new_w0 = np.array(x_novel_w0).flatten().tolist()
            #print("x_new_x0 look the following", self.x_new_w0)
            #print("the size of the x_new_x0",np.size(self.x_new_w0))
            #print("The self.u_new passed to the solver as initial guess u:", self.u_new)

        # After the loop, store the final desired state for completeness
        x_desired_final = self.QuadrotorReferenceTrajectory(self.Duration_simulation, trajectory_type, self.Duration_simulation, self.speed_simulation).flatten()  # Flatten the array
        xDesiredHistory[-1, :] = x_desired_final

        return xHistory, uHistory, xDesiredHistory

    def plot(self, trajectory_type='go_to_point'):

        # Time arrays for states and controls
        tStates = np.linspace(0, self.T, len(xHistory))
        tControls = np.linspace(0, self.T, len(uHistory))

        print("uHistory", uHistory)
        print("xHistory", xHistory)

        # Names of the states and controls for labeling
        state_names = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'xdot', 'ydot', 'zdot', 'psidot']
        control_names = ['u1', 'u2', 'u3', 'u4']

        plt.figure(figsize=(15, 10))  # Adjust the figure size as needed

        # Plot all states
        for i, name in enumerate(state_names):
            plt.subplot(4, 3, i+1)  # Adjust subplot grid as needed based on number of states
            plt.plot(tStates, xHistory[:, i], label='Actual ' + name)
            plt.plot(tStates, xDesiredHistory[:, i], '--', label='Desired ' + name)  # Corrected indexing
            plt.xlabel('Time (s)')
            plt.ylabel(name)
            plt.title(name)
            plt.legend()
            plt.grid(True)

        plt.figure(figsize=(15, 8))  # Adjust the figure size as needed

        # Plot all control inputs
        for i, name in enumerate(control_names):
            plt.subplot(2, 2, i+1)  # Adjust subplot grid as needed based on number of controls
            plt.step(tControls, uHistory[:, i], where='post', label=name)
            plt.xlabel('Time (s)')
            plt.ylabel(name)
            plt.title(f'Control Input: {name}')
            plt.legend()
            plt.grid(True)

        def plot_sphere(ax, center, radius, color='b', alpha=0.3):
            """
            Plots a sphere in a 3D plot.

            :param ax: The axis object to plot on.
            :param center: The center of the sphere (x, y, z).
            :param radius: The radius of the sphere.
            :param color: The color of the sphere.
            :param alpha: Transparency of the sphere.
            """
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_surface(x, y, z, color=color, alpha=alpha)

        # Create a 3D plot for x, y, z positions
        fig3 = plt.figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.plot(xHistory[:, 0], xHistory[:, 1], xHistory[:, 2], label='Actual Position')
        ax3.plot(xDesiredHistory[:, 0], xDesiredHistory[:, 1], xDesiredHistory[:, 2], label='Desired Position', linestyle='--')


        if options.use_box:
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            # Define the vertices of the box
            box_width = 2.5
            box_depth = 2.5
            box_height = 3.0
            box_vertices = np.array([
                [0, 0, 0],
                [0, box_width, 0],
                [box_depth, box_width, 0],
                [box_depth, 0, 0],
                [0, 0, box_height],
                [0, box_width, box_height],
                [box_depth, box_width, box_height],
                [box_depth, 0, box_height]
            ])

            # Define the sides of the box (each side as a list of vertices)
            box_faces = [
                [box_vertices[0], box_vertices[1], box_vertices[5], box_vertices[4]],
                [box_vertices[7], box_vertices[6], box_vertices[2], box_vertices[3]],
                [box_vertices[0], box_vertices[3], box_vertices[7], box_vertices[4]],
                [box_vertices[1], box_vertices[2], box_vertices[6], box_vertices[5]],
                [box_vertices[4], box_vertices[5], box_vertices[6], box_vertices[7]],
                [box_vertices[0], box_vertices[1], box_vertices[2], box_vertices[3]]
            ]

            # Create a Poly3DCollection object
            box = Poly3DCollection(box_faces, facecolors='yellow', linewidths=1, edgecolors='r', alpha=.25)
            ax3.add_collection3d(box)

        def plot_sphere_shell(ax, center, inner_radius, outer_radius, outer_color='#FFA500', inner_color='r', outer_alpha=0.3, inner_alpha=0.5):
            """
            Plots a hollow sphere shell in a 3D plot.

            :param ax: The axis object to plot on.
            :param center: The center of the sphere (x, y, z).
            :param inner_radius: The inner radius of the sphere shell.
            :param outer_radius: The outer radius of the sphere shell.
            :param outer_color: The color of the outer sphere shell.
            :param inner_color: The color of the inner sphere shell.
            :param outer_alpha: Transparency of the outer sphere shell.
            :param inner_alpha: Transparency of the inner sphere shell.
            """
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_outer = center[0] + outer_radius * np.cos(u) * np.sin(v)
            y_outer = center[1] + outer_radius * np.sin(u) * np.sin(v)
            z_outer = center[2] + outer_radius * np.cos(v)
            
            x_inner = center[0] + inner_radius * np.cos(u) * np.sin(v)
            y_inner = center[1] + inner_radius * np.sin(u) * np.sin(v)
            z_inner = center[2] + inner_radius * np.cos(v)
            
            ax.plot_surface(x_inner, y_inner, z_inner, color=inner_color, alpha=inner_alpha)  # Inner surface to create the hollow effect
            ax.plot_surface(x_outer, y_outer, z_outer, color=outer_color, alpha=outer_alpha)

        # Plot the safety distance
        for obstacle in self.obstacles:
            plot_sphere_shell(ax3, obstacle['center'], obstacle['radius'], obstacle['radius'] + self.safety_distance, outer_color='#FFA500', inner_color='r', outer_alpha=0.3, inner_alpha=0.5)

        # Choose intervals at which to display timestamps
        timestamp_interval = len(xHistory) // 10  # For example, label every 10th point
        for i in range(0, len(xHistory), timestamp_interval):
            # Annotate the actual position
            ax3.text(xHistory[i, 0], xHistory[i, 1], xHistory[i, 2], f'{tStates[i]:.1f}', color='blue')
            # Annotate the desired position
            ax3.text(xDesiredHistory[i, 0], xDesiredHistory[i, 1], xDesiredHistory[i, 2], f'{tStates[i]:.1f}', color='red')

        # Plot each obstacle
        for obstacle in self.obstacles:
            plot_sphere(ax3, obstacle['center'], obstacle['radius'], color=obstacle.get('color', 'r'), alpha=0.5)

        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Quadrotor 3D Position with Multiple Obstacles')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Create the parser
    parser = OptionParser()
    # Add arguments
    parser.add_option("", "--trajectory_type", action="store", dest="trajectory_type", type="str", default='sinusoidal',
                        help='Type of trajectory to follow (sinusoidal, long_spline, step_xyz, spline)')
    parser.add_option("--use_nmpc", action="store_true", dest="use_nmpc", default=False,
                    help="Flag to activate NMPC problem creation.")
    parser.add_option("--use_box", action="store_true", dest="use_box", default=False,
                    help="Flag to activate box flight area creation.")

    # Parse the command-line arguments
    (options, args) = parser.parse_args()

    initial_thrust = 38.5
    initial_speed = 0.35# 35
    #controller = MPCController()
    controller = MPCController(initial_speed, initial_thrust, trajectory_type=options.trajectory_type, use_nmpc=options.use_nmpc)

    if options.use_nmpc:
        controller.create_nmpc_problem()

    print(f"Trajectory Type: {options.trajectory_type}")
    print(f"Use NMPC: {options.use_nmpc}")

    xHistory, uHistory, xDesiredHistory = controller.closed_loop_simulation(trajectory_type=options.trajectory_type)
    controller.plot()
