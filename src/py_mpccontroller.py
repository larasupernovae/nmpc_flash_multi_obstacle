#!/usr/bin/env python3

import math
import numpy as np
#import mpctools as mpc
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import casadi as ca
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from optparse import OptionParser

class MPCController():
    def __init__(self, dt = 0.08):
        self.n_states = 10
        self.n_controls = 4
        self.N = 24

        # Define symbolic variables
        self.X = ca.SX.sym('X', self.n_states, self.N+1)  # State trajectory
        self.U = ca.SX.sym('U', self.n_controls, self.N)  # Control trajectory
        self.P = ca.SX.sym('P', self.n_states * (self.N + 1))  # Parameters (reference state trajectory)
        #print("Size of P", np.size(P))

        self.dt = dt #0.08  # Time step
        self.mass = 2.895
        self.gravity = 9.81
        self.mg = self.mass * self.gravity  # mass times gravity
        self.I_xx = (1.0 / 12.0) * self.mass * (0.15 * 0.15 + 0.15 * 0.15)  # (1.0 / 12.0) * mass * (height * height + depth * depth)
        self.I_yy = self.I_xx      # Inertia around y-axis
        self.I_zz = 2*self.I_xx      # Inertia around z-axis

        self.k = 1.0  # Lift constant
        self.l = 0.25  # Distance between rotor and COM
        self.b = 0.2   # Drag constant
        
        self.D = np.array([[0.1, 0, 0],
                           [0, 0.1, 0],
                           [0, 0, 0.1]])

        self.T_hat = np.array([[0, self.mg, 0],
                               [-self.mg, 0, 0],
                               [0, 0, 0]])

        self.I_inv = np.array([[1.0 / self.I_xx, 0, 0],
                               [0, 1.0 / self.I_yy, 0],
                               [0, 0, 1.0 / self.I_zz]])

        self.I = np.array([[self.I_xx, 0, 0],
                           [0, self.I_yy, 0],
                           [0, 0, self.I_zz]])

        # The control parameters for the dynamics 
        self.roll_tau = 0.253
        self.pitch_tau = 0.267
        self.roll_gain = 1.101
        self.pitch_gain = 1.497
        self.K_yaw = 1.8

        # Control input limits
        self.max_thrust = 1.0
        self.max_roll = np.math.radians(20.0)
        self.max_pitch = np.math.radians(20.0)
        self.max_pitch_rate = 5.0/6.0*np.pi     # According to doc
        self.max_roll_rate = 5.0/6.0*np.pi      # According to doc
        self.max_yaw_rate = 5.0/6.0*np.pi       # According to doc
        self.min_thrust = -1.0                   # 37 it hovers...0-80
        self.min_roll = -np.math.radians(20.0)
        self.min_pitch = -np.math.radians(20.0)
        self.min_pitch_rate = -self.max_pitch_rate
        self.min_roll_rate = -self.max_roll_rate
        self.min_yaw_rate = -self.max_yaw_rate

        print("max_thrust", self.max_thrust)
        print("max_roll", self.max_roll)
        print("max_pitch", self.max_pitch)
        print("max_yaw_rate", self.max_yaw_rate)
        # Combine the limits into arrays for easier handling
        self.u_max = np.array([self.max_thrust, self.max_roll , self.max_pitch, self.max_yaw_rate])
        self.u_min = np.array([self.min_thrust, self.min_roll , self.min_pitch, self.min_yaw_rate])
        #u_max = np.array([np.inf, np.inf, np.inf, np.inf])
        #u_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf])


        # Objective function and constraints
        self.Q = ca.DM.eye(self.n_states)  # State cost matrix
        self.R = ca.DM.eye(self.n_controls)  # Control cost matrix
        self.obj = 0  # Objective function
        self.g = []  # Constraints vector

        # Modified weights for Q matrix
        q_x, q_y, q_z = 1.0, 1.0, 1.0  # Increased positional weights
        q_vx, q_vy, q_vz = 0.1, 0.1, 0.1  # Kept the same for velocity
        q_phi, q_theta, q_psi = 0.6, 0.6, 0.1   # Orientation may not be as critical
        q_r = 1.0     # Angular rates are kept the same

        # Modified weights for R matrix
        r_T = 0.03  # Reduced weight for thrust
        r_phi, r_theta, r_psi = 0.1, 0.1, 0.1  # Kept the same for other controls
        
        # Modified weights for Q_final matrix
        q_x_fin, q_y_fin, q_z_fin = 1.0, 1.0, 1.0  # Increased final positional weights
        q_vx_fin, q_vy_fin, q_vz_fin = 0.1, 0.1, 0.1  # Kept the same for final velocity
        q_phi_fin, q_theta_fin, q_psi_fin = 0.6, 0.6, 0.1  # Final orientation may not be as critical
        q_r_fin = 1.0     # Angular rates are kept the same
        
        # Add new weight matrices for MV targets and MV rate of change
        self.R_mv_target = np.diag([0.1, 0.1, 0.1, 0.1])  # Tuning weight for MV target
        self.R_mv_rate = np.diag([0.1, 0.1, 0.1, 0.1])    # Tuning weight for MV rate of change
        
        # Objective function and constraints
        self.Q = np.diag([q_x, q_y, q_z, q_phi, q_theta, q_psi, q_vx, q_vy, q_vz, q_r])
        self.R = np.diag([r_T, r_phi, r_theta, r_psi])
        self.Q_final = np.diag([q_x_fin, q_y_fin, q_z_fin, q_phi_fin, q_theta_fin, q_psi_fin, q_vx_fin, q_vy_fin, q_vz_fin, q_r_fin])
        
        # Nominal control that keeps the quadrotor floating - reference control
        #mv_target = [6.9, 4.9, 4.9, 4.9]
        # self.mv_target = [self.gravity*self.mass, 0.0, 0.0, 0.0]
        self.mv_target = [0.0, 0.0, 0.0, 0.0]
        self.u_last = self.mv_target
        #print("Matrix Q:\n", Q)
        #print("Matrix R:\n", R)
        #print("Matrix Q_final:\n", Q_final)
        
        # Formulate the NMPC problem
        # Modify the NMPC problem formulation

        self.current_time = 0.0
        self.current_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def reset(self):
        self.current_time = 0.0
        self.u_last = self.mv_target

    def notify_velocity(self, vx, vy, vz):
        print("VELOCITY:", vx, vy, vz)
        self.current_state[6] = vx
        self.current_state[7] = vy
        self.current_state[8] = vz
    
    def notify_position(self, x, y, z):
        print("POSITION:", x, y, z)
        self.current_state[0] = x
        self.current_state[1] = y
        self.current_state[2] = z

    def notify_angles(self, roll, pitch, yaw):
        print(f"ANGLES: {math.degrees(roll), math.degrees(pitch), math.degrees(yaw)}")
        self.current_state[3] = roll
        self.current_state[4] = pitch
        self.current_state[5] = yaw
        
    def notify_attitude(self, x, y, z, w):
        pass

    def notify_angle_rates(self, x, y, z):
        self.current_state[9] = z

    def notify_yaw_rate(self, value):
        self.current_state[9] = value

    def drone_dynamics(self, x, u):
        # Extract states   - 10 STATES x,y,z, roll, pitch, yaw, xdot, ydot, zdot, yaw_rate
        pos = x[0:3]  # Position: x, y, z
        angles = x[3:6]  # Orientation: phi, theta, psi
        vel = x[6:9]  # Linear velocities: xdot, ydot, zdot
        ang_vel = x[9]  # Angular velocities: psidot - yaw_rate

        # Control inputs
        thrust = u[0]    # Total thrust
        roll_ref = u[1]  # Roll angle in radians
        pitch_ref = u[2]  # Pitch angle in radians
        yaw_rate_ref = u[3]  # Yaw rate in radians per second

        # thrust *= 0.1
        
        # Rotation matrix from body frame to inertial frame
        ROT = self.rotation_matrix(angles)

        # Translational dynamics
        # Drag coefficients (example values, adjust as needed)
        b_x = 0.1  # Drag coefficient in x direction
        b_y = 0.1  # Drag coefficient in y direction
        b_z = 0.1  # Drag coefficient in z direction

        # Drag acceleration in body frame
        drag_acc_intertial = -ca.vertcat(b_x * vel[0], b_y * vel[1], b_z * vel[2])

        # Gravity vector in inertial frame
        gravity_vector = ca.vertcat(0, 0, -self.gravity)

        # Thrust vector in body frame
        thrust_vector = ca.vertcat(0, 0, thrust)
    
        # Translational dynamics
        #f_com = -gravity * np.array([0, 0, 1]) + np.dot(R, np.array([0, 0, thrust])) / mass + drag_acc + external_forces
        # Force in COM frame
        # translational_dynamics = gravity_vector + ca.mtimes(ROT, thrust_vector) / self.mass # + ca.mtimes(ROT, drag_acc_intertial)
        ## translational_dynamics = ca.vertcat(0, 0, thrust)
        translational_dynamics = ca.vertcat(0, 0, 0)
        # translational_dynamics = gravity_vector + thrust_vector / self.mass # + ca.mtimes(ROT, drag_acc_intertial)
        
        #### translational_dynamics = gravity_vector
        # Rotational dynamics
        # Assuming simple proportional control for roll and pitch, and direct yaw rate control
        droll = (1 / self.roll_tau) * (self.roll_gain*(roll_ref - angles[0]))
        dpitch = (1 / self.pitch_tau) * (self.pitch_gain*(pitch_ref - angles[1]))
        dyaw = angles[2]
    
        ddyaw = - self.K_yaw * (yaw_rate_ref - ang_vel)

        # Acceleration in xy is given by the current angles
        factor = -10.0
        ddx = (1 / self.roll_tau) * angles[1]*factor    # Assuming no rotation
        ddy = (1 / self.pitch_tau) * angles[0]*factor  # Assuming no rotation
        ddz = 0.0
        
        ##        vel[2] = (thrust-37.5)*14.6/7.0
        ## vel[2] = (thrust-37.5)*2*thrust
        vel[2] = thrust*30.0
    
        rotational_dynamics = ca.vertcat(droll, dpitch, dyaw)
        translational_dynamics = ca.vertcat(ddx, ddy, ddz)
        # linear_velocities, derivations of the angles, acceleration of the drones body frame thrust/drag
        # Combine the dynamics
        # f = ca.vertcat(vel, rotational_dynamics, translational_dynamics, ddyaw)
        f = ca.vertcat(vel, rotational_dynamics, translational_dynamics, ddyaw)

        # The state vector x is ordered as [orientation angles, linear velocities, angular velocities].
        # The output vector f is ordered as [linear velocities, derivatives of orientation angles
        # (rotational dynamics), derivatives of linear velocities (translational dynamics)].
        return f

    # Rotation matrix function
    def rotation_matrix(self, angles):
        phi, theta, psi = angles[0], angles[1], angles[2]
        # Rotation matrices for Rz, Ry, Rx
        Rz = ca.vertcat(ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
                        ca.horzcat(ca.sin(psi), ca.cos(psi), 0),
                        ca.horzcat(0, 0, 1))
        Ry = ca.vertcat(ca.horzcat(ca.cos(theta), 0, ca.sin(theta)),
                        ca.horzcat(0, 1, 0),
                        ca.horzcat(-ca.sin(theta), 0, ca.cos(theta)))
        Rx = ca.vertcat(ca.horzcat(1, 0, 0),
                        ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
                        ca.horzcat(0, ca.sin(phi), ca.cos(phi)))
        R_dyn = ca.mtimes(Rz, ca.mtimes(Ry, Rx))
        return R_dyn
    

    def create_nmpc_problem(self):
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            # Objective function
            self.obj += ca.mtimes([(st - self.P[self.n_states*k:self.n_states*(k+1)]).T, self.Q, st - self.P[self.n_states*k:self.n_states*(k+1)]])  # State cost
            self.obj += ca.mtimes([con.T, self.R, con])  # Control cost
            # Control cost, penalizing deviation from mv_target
            self.obj += ca.mtimes([(con - self.mv_target).T, self.R_mv_target, (con - self.mv_target)])  # MV target deviation cost

            # For the rate of change, compare to the previous control input if not the first step
            if k > 0:
                con_prev = self.U[:, k-1]
                self.obj += ca.mtimes([(con - con_prev).T, self.R_mv_rate, (con - con_prev)])  # MV rate of change cost

            st_next = self.X[:, k+1]
            f = self.drone_dynamics(st, con)

            #A, B = drone_dynamics_jacobians(st, con)
    
            # Linearize the dynamics around the current state and control
            # Linearize the dynamics around the current state and control
            # This is an approximation for small deviations
            #if k < N - 1:
            #    u_next = U[:, k+1]
            #else:
            #   u_next = U[:, k]  # No future control input, use current control input

            # This is an approximation for small deviations
            #linearized_dynamics = f + ca.mtimes(A, st_next - st) + ca.mtimes(B, u_next - con)
            # System dynamics constraint using linearized dynamics
            #dynamic_constraint = st_next - (st + dt * linearized_dynamics)
            #g.append(dynamic_constraint)

            dynamic_constraint = st_next - (st + self.dt * f)  # System dynamics constraint
            self.g.append(dynamic_constraint)
            ###print("Velicina dynamics constraints: ", np.size(dynamic_constraint))
            # Print the dynamic constraint for this time step
            #print(f"Dynamics constraint at step {k}: {dynamic_constraint}")

        # Final state cost
        self.obj += ca.mtimes([(self.X[:, self.N] - self.P[self.n_states*self.N:self.n_states*(self.N+1)]).T, self.Q_final, self.X[:, self.N] - self.P[self.n_states*self.N:self.n_states*(self.N+1)]])

        opts = {'ipopt.print_level': 0, 'print_time': 0, #3 1
                'ipopt.max_iter': 3000,  # Maximum number of iterations
                'ipopt.tol': 1e-10, #2,       # Tolerance for convergence the bigger the number the tighter the tolerance
                'ipopt.acceptable_tol': 1e-12, #3, # Acceptable tolerance for convergence
                'ipopt.linear_solver': 'mumps'# 'mumps' #'ma27 reko bjorn' # Linear solver to be used}  # Increased print level
                }

        # Since we have no constraints on the states, the only constraints are the dynamics
        nlp = {
            'x': ca.vertcat(ca.reshape(self.U, -1, 1), ca.reshape(self.X, -1, 1)),
            'f': self.obj,  # objective function
            'g': ca.vertcat(*self.g),   # constraints
            'p': self.P        # refernce trajectory
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)   #sqs

        #print("Objective function:\n", obj)
        #print("Constraints:\n", g)

        # Define bounds for decision variables (control inputs and states)
        self.lbx = np.concatenate([np.tile(self.u_min, self.N), -np.inf* np.ones(self.n_states * (self.N + 1))])
        self.ubx = np.concatenate([np.tile(self.u_max, self.N), np.inf* np.ones(self.n_states * (self.N + 1))])  # np.inf * np.ones(n_states * (N + 1))])
        ###print("Velicina lbx: ", np.size(self.lbx))
        ###print("Velicina ubx: ", np.size(self.ubx))
        # Define bounds for constraints (only dynamic constraints)
        self.lbg = np.zeros(self.n_states * self.N)  # Dynamics are equality constraints; for equality constraints.
        self.ubg = np.zeros(self.n_states * self.N)
        #print("Velicina lbg: ", np.size(self.lbg))
        #print("Velicina ubg: ", np.size(self.ubg))


    def solve_nmpc(self, xk, x_ref_full, mv_target, u_last):

        # Ensure x_ref_full is a 1D array
        x_ref_full_flat = np.reshape(x_ref_full, -1)
        #print("x_ref_full_flat:", x_ref_full_flat)
        # Concatenate the flattened x0 with x_ref_full
        # p = np.concatenate([xk_flat, x_ref_full_flat])

        # Define a neutral initial guess for control inputs if mv_target is not suitable
        #neutral_control_input = [0, 0, 0, 0]  # Example neutral control input
        #u0_repeated = np.tile(neutral_control_input, N).flatten()

        # Use the last optimal control inputs as the initial guess if available
        if u_last is not None:
            u0_repeated = np.tile(u_last, self.N).flatten()
        else:
            neutral_control_input = [0, 0, 0, 0]  # Neutral control input
            u0_repeated = np.tile(neutral_control_input, self.N).flatten()

        # Initial guess for state variables, repeated for each time step
        x0_repeated = np.tile(xk, (self.N + 1, 1)).flatten()

        # Concatenate initial guesses for control inputs and state variables
        initial_guess = np.concatenate([u0_repeated, x0_repeated])
        #print("initial_guess:", initial_guess)
   
        # Set up the arguments for the solver
        args = {
            'x0': initial_guess,   # initial guess
            'lbx': self.lbx,  # Bounds for control inputs and states
            'ubx': self.ubx,
            'lbg': self.lbg,  # Bounds for dynamic constraints
            'ubg': self.ubg,
            'p': x_ref_full_flat
        }

        # Print the arguments before passing them to the solver
        #print("Arguments passed to the solver:")
        #for key, value in args.items():
        #    print(f"{key}: {value}")

        #print("Types of bounds:", type(lbx), type(ubx), type(lbg), type(ubg))
        #print("Lengths of bounds:", len(lbx), len(ubx), len(lbg), len(ubg))

        sol = self.solver(**args)
        ### print("Solver return keys:", sol.keys())  # Print the keys of the solution dictionary
        if self.solver.stats()['success']:
            ### print("Solver succeeded.")

            # Extract the full solution
            full_solution = np.array(sol['x'])
            #print("Full solution shape:", full_solution.shape)
            #print("Full solution:", full_solution)

            # Extract control inputs and states from the full solution
            #u_opt = full_solution[:n_controls*N].reshape(n_controls, N)
            u_opt = full_solution[:self.n_controls*self.N].reshape(self.N, self.n_controls)
            # Extract state variables
            #x_opt = full_solution[n_controls*N:].reshape(n_states, N+1)
            x_opt = full_solution[self.n_controls*self.N:].reshape(self.N+1, self.n_states)
            #print("Extracted state variables shape:", x_opt.shape)
            #print("First state variables:", x_opt[:, 0])
            
            #print("Optimal control inputs:", u_opt[:,0])
            return x_opt, u_opt
        else:
            print("Solver failed:", self.solver.stats()['return_status'])
            return None, None

    def QuadrotorReferenceTrajectory1(self, t, duration=20.0, num_points=5):
        # Define the 5 points in XYZ space for the spline
        # These points can be adjusted as per requirement
        points = np.array([
            [0.0, 0.0, 1.0],  # Start point
            [2.0, 2.0, 2.0],  # Example point
            [4.0, -2.0, 3.0], # Example point
            [6.0, 3.0, 2.0],  # Example point
            [8.0, 0.0, 1.0]   # End point
        ])

        # Time points for the spline (equally spaced)
        time_points = np.linspace(0, duration, num_points)

        # Create a spline interpolation for each dimension
        spline_x = interp1d(time_points, points[:, 0], kind='cubic')
        spline_y = interp1d(time_points, points[:, 1], kind='cubic')
        spline_z = interp1d(time_points, points[:, 2], kind='cubic')

        # Compute the position at time t using the spline
        x = spline_x(np.clip(t, 0, duration))
        y = spline_y(np.clip(t, 0, duration))
        z = spline_z(np.clip(t, 0, duration))

        # Initialize other states to zero
        phi = theta = psi = xdot = ydot = zdot = psidot = np.zeros_like(t)

        # Combine all states into a single array
        xdesired = np.vstack([x, y, z, phi, theta, psi, xdot, ydot, zdot, psidot])
        
        return xdesired

    # Define a function to generate the reference trajectory (similar to MATLAB)
    def QuadrotorReferenceTrajectory2(self, t):
        # Calculate the desired trajectory
        x = 6 * np.sin(t / 3)
        y = -6 * np.sin(t / 3) * np.cos(t / 3)
        z = 6 * np.cos(t / 3)

        # Initialize other states to zero
        phi = np.zeros_like(t)
        theta = np.zeros_like(t)
        psi = np.zeros_like(t)
        xdot = np.zeros_like(t)
        ydot = np.zeros_like(t)
        zdot = np.zeros_like(t)
        
        psidot = np.zeros_like(t)
        # Combine all states into a single array
        xdesired = np.array([x, y, z, phi, theta, psi, xdot, ydot, zdot, psidot])

        return xdesired

    def QuadrotorReferenceTrajectory3(self, t):
        # Define the initial position and velocity
        x0, y0, z0 = 0.0, 0.0, 0.0  # Initial position
        v = 5.0  # Velocity
        vz = 0.20

        x = 0.0
        z = 0.0
        if t >= 5.0 and  t<= 15.0:
            x = x0 + v * (t-5.0)
            xdot = v
            z = z0 + vz*(t-5.0)
            zdot = vz
        else:
            xdot = 0.0
            zdot = 0.0
            if t> 15.0:
                x = x0 + v*10.0
                z = z0 + vz*10.0

        y = 0.0
        # z = 0.0
        phi = 0.0
        theta = 0.0
        psi = 0.0
        ydot = 0.0
        # zdot = 0.0
        psidot = 0.0

        xdesired = np.vstack([x, y, z, phi, theta, psi, xdot, ydot, zdot, psidot])

        return xdesired
        
        # Calculate the desired trajectory
        x = np.where(t >= 5.0 and t < 15.0, x0 + v * (t-5.0), x0 + v * 10.0)
        # x = np.where(t < 10.0, x0 + v * t, x0 + v * 10.0)
        y = np.where(t < 10.0, y0 + v/2.0 * t, y0 + v/2.0 * 10.0)
        z = np.where(t < 10.0, z0 + v/3.0 * t, z0 + v/3.0 * 10.0)
        ## z = z0 * np.ones_like(t)

        #for i in range(len(x)):
        #    if t[i] < 5.0:
        #        x[i] = 0.0
        
        # xdot = np.where(t >=5.0 and t < 15.0, v, 0.0)
        xdot = np.where(t < 10.0, v, 0.0)
        ydot = np.where(t < 10.0, v/2.0, 0.0)
        zdot = np.where(t < 10.0, v/3.0, 0.0)
        # zdot = np.zeros_like(t)

        ## x = np.zeros_like(t)
        y = np.zeros_like(t)
        
        ## xdot = np.zeros_like(t)
        ydot = np.zeros_like(t)

        # Initialize other states to zero
        phi = np.zeros_like(t)
        theta = np.zeros_like(t)
        psi = np.zeros_like(t)
        
        psidot = np.zeros_like(t)
        
        # Combine all states into a single array
        xdesired = np.vstack([x, y, z, phi, theta, psi, xdot, ydot, zdot, psidot])

        return xdesired

    # Define the drone's dynamics as a nonlinear function
    def drone_dynamics_numerical(self, x, u):
        # Extract states
        pos = x[0:3]  # Position: x, y, z
        angles = x[3:6]  # Orientation: phi, theta, psi
        vel = x[6:9]  # Linear velocities: xdot, ydot, zdot
        ang_vel = x[9]  # Angular velocities: psidot - yaw_rate

        ### print("VELOCITY:", vel)

        # Control inputs
        thrust = u[0]  # Total thrust
        roll_ref = u[1]  # Roll angle in radians
        pitch_ref = u[2]  # Pitch angle in radians
        yaw_rate_ref = u[3]  # Yaw rate in radians per second

        # thrust *= 0.1
        
        # Rotation matrix from body frame to inertial frame
        ROT = self.rotation_matrix_numerical(angles)


        # Translational dynamics
        # Drag coefficients (example values, adjust as needed)
        b_x = 0.1  # Drag coefficient in x direction
        b_y = 0.1  # Drag coefficient in y direction
        b_z = 0.1  # Drag coefficient in z direction

        # Drag acceleration in body frame
        drag_acc_intertial = -ca.vertcat(b_x * vel[0], b_y * vel[1], b_z * vel[2])
        #print("Size of drag_acc_intertial", np.size(drag_acc_intertial))

        # Gravity vector in inertial frame
        gravity_vector = ca.vertcat(0, 0, -self.gravity)
        #print("Size of gravity_vector", np.size(gravity_vector))
        
        # Thrust vector in body frame
        ### thrust_vector = ca.vertcat(0, 0, thrust)
        thrust_vector = ca.vertcat(0, 0, thrust)
        #print("Size of thrust_vector", np.size(thrust_vector))

        #f_com = -gravity * np.array([0, 0, 1]) + np.dot(R, np.array([0, 0, thrust])) / mass + drag_acc + external_forces
        # Force in COM frame
        # f_com = gravity_vector + ca.mtimes(ROT, thrust_vector) / self.mass # + ca.mtimes(ROT, drag_acc_intertial)
        # f_com = gravity_vector +  thrust_vector
        f_com = ca.vertcat(0, 0, 0)
        ### f_com = gravity_vector
        # Convert CasADi DM to NumPy array for concatenation
        f_com_np = np.array(f_com).flatten()  # Flatten to ensure it's 1D

        #print("Size of f_com", np.size(f_com))
        # Rotational dynamics
        # Assuming simple proportional control for roll and pitch, and direct yaw rate control
        droll = (1 / self.roll_tau) * (self.roll_gain*(roll_ref - angles[0]))
        dpitch = (1 / self.pitch_tau) * (self.pitch_gain*(pitch_ref - angles[1]))
        dyaw =  angles[2]
        
        ddyaw = -self.K_yaw * (yaw_rate_ref - ang_vel)

        factor = -13.0
        ddx = (1 / self.roll_tau) * angles[1]*factor    # Assuming no rotation
        ddy = (1 / self.pitch_tau) * angles[0]*factor  # Assuming no rotation
        ddz = 0.0

        vel[2] = thrust*30.0
        # vel[2] = (thrust-37.5)*14.6/7.0        
        # Combine the dynamics
        # print("f_com_np***************''", f_com_np)
        ## f = np.concatenate((vel, np.array([droll, dpitch, dyaw]), f_com_np, np.array([ddyaw])))
        f = np.concatenate((vel, np.array([droll, dpitch, dyaw]), np.array([ddx, ddy, ddz]), np.array([ddyaw])))
        
        #print("Size of f", np.size(f))
        return f


    def rotation_matrix_numerical(self, angles):
        phi, theta, psi = angles
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0],
                       [0,            0,           1]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0,            1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0,           0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi),  np.cos(phi)]])
        return np.dot(Rz, np.dot(Ry, Rx))


    def get_trajectory(self, traj):
        reftraj = None
        if traj==1:
            reftraj = self.QuadrotorReferenceTrajectory1
        if traj==2:
            reftraj = self.QuadrotorReferenceTrajectory2
        if traj==3:
            reftraj = self.QuadrotorReferenceTrajectory3
        return reftraj

    def tick(self, traj=1):
        reftraj = self.get_trajectory(traj)
        yref = np.array([reftraj(self.current_time) for j in range(self.N+1)]).T
        x_opt, u_opt = self.solve_nmpc(self.current_state, yref.T, self.mv_target, self.u_last)
        if x_opt is not None and u_opt is not None:
            self.u_last = u_opt[0, :self.n_controls]
            x_current = x_opt[0, :self.n_states]
            ###print(f"Time: {self.current_time:.2f}, Optimal Control: {self.u_last}")
            ###print(f"Time: {self.current_time:.2f}, Optimal Position: {x_current}")
        self.current_time += self.dt
        return self.u_last, yref
            
    def closed_loop_simulation(self, Duration, traj=1): 
        # MOJE POKUSAVAS
        # Closed-loop simulation

        Ts = self.dt
        reftraj = self.get_trajectory(traj)



        # Define the initial conditions (match MATLAB)
        x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Simulation parameters
        N_sim = int(Duration / Ts)  # Number of simulation steps
        # Example time array
        #desired_trajectory = QuadrotorReferenceTrajectory(t)
        
        self.xHistory = np.zeros((N_sim+1, self.n_states))
        self.uHistory = np.zeros((N_sim, self.n_controls))
        self.xHistory[0, :] = x0  # Initial state
        u_last = self.mv_target # None  # Initialize the last optimal control input


        for k in range(N_sim):
            # Current time
            current_time = k * Ts
            t = np.linspace(current_time, (k+self.N)*Ts, self.N)
            # Generate reference trajectory for each step in the prediction horizon
            yref = np.array([reftraj(current_time + j * Ts) for j in range(self.N+1)]).T.squeeze()
            #times = [current_time + j * Ts for j in range(self.N+1)]
            #yref = reftraj(times)
            print("YREF:", k, yref.T)
            #yref = QuadrotorReferenceTrajectory(t)

            # Get current state
            x_current = self.xHistory[k, :]

            ### print("X-CURRRENT:", x_current, yref.T, self.mv_target, u_last)            
            # Compute the control moves with NMPC
            x_opt, u_opt = self.solve_nmpc(x_current, yref.T, self.mv_target, u_last)
    
            #print(f"Time: {current_time:.2f}, Optimal Position from solver: {x_opt}")
            #print(f"Time: {current_time:.2f}, Optimal Control from solver: {u_opt}")
            if x_opt is not None and u_opt is not None:
                # Update histories
                #xHistory[k + 1, :] = x_opt[0, :n_states]
                self.uHistory[k, :] = u_opt[0, :self.n_controls]
                ### x_current = x_opt[0, :self.n_states]
                # Update last control
                u_last = u_opt[0, :self.n_controls]

                # Print information (match formatting with MATLAB)
                print(f"Time: {current_time:.2f}, Optimal Control: {u_last}")
                print(f"Time: {current_time:.2f}, Optimal Position: {x_current}")

            try:
                # State update using numerical integration with the drone dynamics
                ####print("X-CURRRENT:", x_current)
                ode_result = solve_ivp(lambda t, y: self.drone_dynamics_numerical(x_current, u_last), 
                                       [current_time, current_time + Ts], x_current,
                                       method='RK45', max_step=Ts/10)  # Example of setting a maximum step size
                if ode_result.status != 0:
                    print(f"Integration was unsuccessful: {ode_result.message}")
                else:
                    x_new = ode_result.y[:, -1]
                    ####print("XNEW:", x_new)
                    self.xHistory[k + 1, :] = x_new
                    # print(f"size of x_new: {np.size(x_new)}")
                    x_current = x_new
            except Exception as e:
                print(f"An error occurred: {e}")

    def plot(self, Duration, traj=1):
        # Plotting (adjust to match MATLAB plots)
        #time = np.arange(0, Duration + Ts, Ts)
        # Generate the time array

        Ts = self.dt

        if traj==1:
            reftraj = self.QuadrotorReferenceTrajectory1
        if traj==2:
            reftraj = self.QuadrotorReferenceTrajectory2
        if traj==3:
            reftraj = self.QuadrotorReferenceTrajectory3

        N_sim = int(Duration / Ts)  # Number of simulation steps
        
        time = np.linspace(0, Duration, N_sim + 1)

        yref_tot = np.array([reftraj(np.array([t_step])) for t_step in time]).squeeze()
        print("YREFTOT:", yref_tot)
        
        # Create subplots for states and control inputs
        # Create subplots for states, control inputs, and velocities
        fig, axs = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Quadrotor States, Control Inputs, and Velocities', fontsize=16)
        
        # Plot the states (positions and angles)
        for i in range(6):
            row = i // 3
            col = i % 3
            axs[row, col].plot(time, self.xHistory[:, i], label='Actual State')
            axs[row, col].plot(time, yref_tot[:, i], label='Reference', linestyle='--')
            axs[row, col].set_ylabel(['x', 'y', 'z', 'phi', 'theta', 'psi'][i])
            axs[row, col].set_title(['Position x', 'Position y', 'Position z', 'Angle phi', 'Angle theta', 'Angle psi'][i])
            axs[row, col].legend()
            axs[row, col].grid(True)
            
            # Plot the velocities
            velocity_labels = ['Vx', 'Vy', 'Vz']
        for i in range(3):
            row = 2  # Third row for velocities
            col = i
            axs[row, col].plot(time, self.xHistory[:, i+6], label='Actual Velocity')  # i+6 to access velocity states
            axs[row, col].plot(time, yref_tot[:, i+6], label='Reference Velocity', linestyle='--')
            axs[row, col].set_ylabel(velocity_labels[i])
            axs[row, col].set_title(f'Velocity {velocity_labels[i]}')
            axs[row, col].legend()
            axs[row, col].grid(True)
            
        # Plot the control inputs
        fig2, axs2 = plt.subplots(2, 2, figsize=(14, 8))
            
        for i in range(self.n_controls):
            row = i // 2
            col = i % 2
            axs2[row, col].plot(time[:-1], self.uHistory[:, i], label=f'Control {i+1}')
            axs2[row, col].set_xlabel('Time')
            axs2[row, col].set_ylabel(f'Control Input {i+1}')
            axs2[row, col].set_title(['Thrust','Roll', 'Pitch', 'Yaw_Rate'][i])
            axs2[row, col].legend()
            axs2[row, col].grid(True)
            
        # Create a 3D plot for x, y, z positions
        fig3 = plt.figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.plot(self.xHistory[:, 0], self.xHistory[:, 1], self.xHistory[:, 2], label='Actual Position')
        ax3.plot(yref_tot[:, 0], yref_tot[:, 1], yref_tot[:, 2], label='Reference Target', linestyle='--')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Quadrotor 3D Position')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


        
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option ("", "--traj", action="store", dest="traj", type="int", help="Trajectory", default=1)    
    (options, args) = parser.parse_args()
    
    controller = MPCController()
    controller.create_nmpc_problem()
    controller.closed_loop_simulation(20, traj=options.traj)
    controller.plot(20, traj=options.traj)
