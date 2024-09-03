#!/usr/bin/env python3
import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import casadi as ca
from scipy.integrate import odeint

# Define the dimensions of the state and control input
n_states = 12  
n_controls = 4 
N = 24  # Horizon length
print("Time horizon:", N)

# Define symbolic variables
X = ca.SX.sym('X', n_states, N+1)  # State trajectory
U = ca.SX.sym('U', n_controls, N)  # Control trajectory
P = ca.SX.sym('P', n_states * (N + 2))  # Parameters (initial state and reference state trajectory)

dt = 0.08  # Time step
roll_gain = 1.101 
pitch_gain = 1.097  
mass = 2.895
roll_time_constant = 0.253
pitch_time_constant = 0.267
b_thrust = 1.0 / mass 
b_roll = 1.0 / roll_time_constant  
b_pitch = 1.0 / pitch_time_constant  
b_yaw = 1.8 
gravity = 9.81
mg = mass * gravity  # mass times gravity
I_xx = 0.1  # inertia around x-axis
I_yy = 0.1  # inertia around y-axis
I_zz = 0.1  # inertia around z-axis

k = 1.0  # Lift constant
l = 0.25  # Distance between rotor and COM
m = 2.0   # Mass
b = 0.2   # Drag constant

D = np.array([[0.4, 0, 0],
              [0, 0.3, 0],
              [0, 0, 0.2]])

T_hat = np.array([[0, mg, 0],
                  [-mg, 0, 0],
                  [0, 0, 0]])

I_inv = np.array([[1.0 / I_xx, 0, 0],
                  [0, 1.0 / I_yy, 0],
                  [0, 0, 1.0 / I_zz]])

I = np.array([[I_xx, 0, 0],
              [0, I_yy, 0],
              [0, 0, I_zz]])

# Control input limits
max_thrust = 80
max_roll = 10
max_pitch = 10
max_yaw = 360.0 * np.pi / 180.0
min_thrust = -20
min_roll = -10
min_pitch = -10
min_yaw = -360.0 * np.pi / 180.0

# Combine the limits into arrays for easier handling
u_max = np.array([max_thrust, max_roll, max_pitch, max_yaw])
u_min = np.array([min_thrust, min_roll, min_pitch, min_yaw])

# System dynamics function (replace this with your system's dynamics)
def dynamics(x, u):
    # Example: x_dot = A * x + B * u
    A = ca.DM([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1-dt/mass * D[0, 0], -dt/mass * D[0, 1], -dt/mass * D[0, 2], dt/mass * T_hat[0, 0], dt/mass * T_hat[0, 1], dt/mass * T_hat[0, 2], 0, 0, 0],
              [0, 0, 0, -dt/mass * D[1, 0], 1-dt/mass * D[1, 1], -dt/mass * D[1, 2], dt/mass * T_hat[1, 0], dt/mass * T_hat[1, 1], dt/mass * T_hat[1, 2], 0, 0, 0],
              [0, 0, 0, -dt/mass * D[2, 0], -dt/mass * D[2, 1], 1-dt/mass * D[2, 2], dt/mass * T_hat[2, 0], dt/mass * T_hat[2, 1], dt/mass * T_hat[2, 2], 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    B = ca.DM([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [dt/mass, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, I_inv[0, 0], I_inv[0, 1], I_inv[0, 2]],
        [0, I_inv[1, 0], I_inv[1, 1], I_inv[1, 2]],
        [0, I_inv[2, 0], I_inv[2, 1], I_inv[2, 2]]                  
    ])
   # print("Matrix A:\n", A)
   # print("Matrix B:\n", B)
    x_dot = ca.mtimes(A, x) + ca.mtimes(B, u)
    return x_dot

# Objective function and constraints
Q = ca.DM.eye(n_states)  # State cost matrix
R = ca.DM.eye(n_controls)  # Control cost matrix
Q_final = ca.DM.eye(n_states)  # State cost matrix final
obj = 0  # Objective function
g = []  # Constraints vector

# Example weights for Q matrix
q_x, q_y, q_z, q_vx, q_vy, q_vz = 0.8, 1.0, 1.0, 1.0, 1.0, 1.0
q_phi, q_theta, q_psi, q_p, q_q, q_r = 0, 0, 0, 0, 0, 0

# Example weights for R matrix
r_T, r_phi, r_theta, r_psi = 0.01, 0.05, 0.05, 0.05

# agressive control 0.01, 0.01, 0.01, 0.01

# Example weights for Q_final matrix
q_x_fin, q_y_fin, q_z_fin, q_vx_fin, q_vy_fin, q_vz_fin = 0.8, 1.0, 1.0, 1.0, 1.0, 1.0
q_phi_fin, q_theta_fin, q_psi_fin, q_p_fin, q_q_fin, q_r_fin = 0, 0, 0, 0, 0, 0

# Objective function and constraints
Q = np.diag([q_x, q_y, q_z, q_vx, q_vy, q_vz, q_phi, q_theta, q_psi, q_p, q_q, q_r])
R = np.diag([r_T, r_phi, r_theta, r_psi])
Q_final = np.diag([q_x_fin, q_y_fin, q_z_fin, q_vx_fin, q_vy_fin, q_vz_fin, q_phi_fin, q_theta_fin, q_psi_fin, q_p_fin, q_q_fin, q_r_fin])

#print("Matrix Q:\n", Q)
#print("Matrix R:\n", R)
#print("Matrix Q_final:\n", Q_final)

# Formulate the NLP
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj += ca.mtimes([(st - P[n_states*k:n_states*(k+1)]).T, Q, st - P[n_states*k:n_states*(k+1)]])  # State cost
    obj += ca.mtimes([con.T, R, con])  # Control cost
    st_next = X[:, k+1]
    f = dynamics(st, con)
    # Add control input constraints and slew rate constraints as per your problem formulation
    dynamic_constraint = st_next - (st + dt * f)  # System dynamics constraint
    g.append(dynamic_constraint)

    # Print the dynamic constraint for this time step
    #print(f"Dynamics constraint at step {k}: {dynamic_constraint}")

# Final state cost
obj += ca.mtimes([(X[:, N] - P[n_states*N:n_states*(N+1)]).T, Q_final, X[:, N] - P[n_states*N:n_states*(N+1)]])

opts = {'ipopt.print_level': 3, 'print_time': 1, 
    'ipopt.max_iter': 1000,  # Maximum number of iterations
    'ipopt.tol': 1e-2,       # Tolerance for convergence
    'ipopt.acceptable_tol': 1e-3, # Acceptable tolerance for convergence
    'ipopt.linear_solver': 'mumps'#'mumps' # Linear solver to be used}  # Increased print level
}

# Since we have no constraints on the states, the only constraints are the dynamics
nlp = {
    'x': ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1)),
    'f': obj,
    'g': ca.vertcat(*g),
    'p': P
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

#print("Objective function:\n", obj)
#print("Constraints:\n", g)

# Define bounds for decision variables (control inputs and states)
lbx = np.concatenate([np.tile(u_min, N), -np.inf * np.ones(n_states * (N + 1))])
ubx = np.concatenate([np.tile(u_max, N), np.inf * np.ones(n_states * (N + 1))])  # np.inf * np.ones(n_states * (N + 1))])

# Define bounds for constraints (only dynamic constraints, no bounds on states)
lbg = np.zeros(n_states * N)  # Dynamics are equality constraints
ubg = np.zeros(n_states * N)

#print("Objective function:\n", obj)
#print("Constraints:\n", g)

def solve_nmpc(x0, xk, x_ref_full, mv_target):

    xk_flat = np.reshape(xk, -1)
    # Ensure x_ref_full is a 1D array
    x_ref_full_flat = np.reshape(x_ref_full, -1)
    # Concatenate the flattened x0 with x_ref_full
    p = np.concatenate([xk_flat, x_ref_full_flat])

    # Repeat x0 for each time step in the prediction horizon
    x0_repeated = np.tile(x0, (N + 1, 1)).flatten()

    # Repeat mv_target for each control interval
    u0_repeated = np.tile(mv_target, N).flatten()

    # Concatenate the repeated x0 and mv_target to form the initial guess
    initial_guess = np.concatenate([x0_repeated, u0_repeated])
    # Set up the arguments for the solver
    args = {
        'x0': initial_guess,   # initial guess
        'p': p,
        'lbx': lbx,  # Bounds for control inputs and states
        'ubx': ubx,
        'lbg': lbg,  # Bounds for dynamic constraints
        'ubg': ubg
    }

    # Print the arguments before passing them to the solver
    #print("Arguments passed to the solver:")
    #for key, value in args.items():
    #    print(f"{key}: {value}")

    #print("Types of bounds:", type(lbx), type(ubx), type(lbg), type(ubg))
    #print("Lengths of bounds:", len(lbx), len(ubx), len(lbg), len(ubg))

    sol = solver(**args)
    print("Solver return keys:", sol.keys())  # Print the keys of the solution dictionary
    if solver.stats()['success']:
        print("Solver succeeded.")

        # Extract the full solution
        full_solution = np.array(sol['x'])
        #print("Full solution shape:", full_solution.shape)
        #print("Full solution:", full_solution)

        # Extract control inputs and states from the full solution
        u_opt = full_solution[:n_controls*N].reshape(n_controls, N)

        # Extract state variables
        x_opt = full_solution[n_controls*N:].reshape(n_states, N+1)
        #print("Extracted state variables shape:", x_opt.shape)
        #print("First state variables:", x_opt[:, 0])
        
        #print("Optimal control inputs:", u_opt[:,0])
        return x_opt, u_opt
    else:
        print("Solver failed:", solver.stats()['return_status'])
        return None, None
    


# Define the initial conditions (match MATLAB)
#x0 = [7,-10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Nominal control that keeps the quadrotor floating (match MATLAB)
mv_target = [4.9, 4.9, 4.9, 4.9]

current_x = 7
current_y = -10
current_z = 0
current_vx = 0
current_vy = 0
current_vz = 0
current_roll = 0
current_pitch = 0
current_yaw = 0
ang_vel_roll = 0
ang_vel_pitch = 0
ang_vel_yaw = 0

# Assuming current_x, current_y, etc., are defined somewhere in your code
x0 = np.array([current_x, current_y, current_z, current_vx, current_vy, current_vz, 
               current_roll, current_pitch, current_yaw, ang_vel_roll, ang_vel_pitch, ang_vel_yaw])
'''
# Define x_ref for each time step in the horizon
x_ref = np.array([target_x, target_y, target_z, target_vx, target_vy, target_vz, 
                  target_roll, target_pitch, target_yaw, target_ang_vel_roll, target_ang_vel_pitch, target_ang_vel_yaw])

# Repeat x_ref for each time step in the horizon and flatten it
x_ref_full = np.tile(x_ref, N + 1)
'''

# Simulation parameters
Duration = 20
Ts = 0.1  # Sample time
N_sim = int(Duration / Ts)  # Number of simulation steps

# Define a function to generate the reference trajectory (similar to MATLAB)
def QuadrotorReferenceTrajectory(t):
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
    phidot = np.zeros_like(t)
    thetadot = np.zeros_like(t)
    psidot = np.zeros_like(t)

    # Combine all states into a single array
    xdesired = np.array([x, y, z, phi, theta, psi, xdot, ydot, zdot, phidot, thetadot, psidot])

    return xdesired


t = np.linspace(0, 20, 200)  # Example time array
desired_trajectory = QuadrotorReferenceTrajectory(t)

# If you need the trajectory at a specific time instant, say t = 5 seconds
t_single = 10
desired_state_at_t = QuadrotorReferenceTrajectory(np.array([t_single]))

# Define the drone's dynamics as a nonlinear function
def drone_dynamics_numerical(x, u):
    # Extract states
    pos = x[0:3]  # Position: x, y, z
    angles = x[3:6]  # Orientation: phi, theta, psi
    vel = x[6:9]  # Linear velocities: xdot, ydot, zdot
    ang_vel = x[9:12]  # Angular velocities: phidot, thetadot, psidot

    # Control inputs: squared angular velocities of rotors
    u1, u2, u3, u4 = u[0], u[1], u[2], u[3]

    # Rotation matrix from body frame to inertial frame
    R = rotation_matrix(angles)

    # Total thrust
    T = k * (u1 + u2 + u3 + u4)

    # Torques in the direction of phi, theta, psi
    tau_beta = ca.vertcat(l * k * (-u2 + u4), l * k * (-u1 + u3), b * (-u1 + u2 - u3 + u4))

    # Translational dynamics
    f_pos = vel
    f_vel = -gravity * ca.vertcat(0, 0, 1) + ca.mtimes(R, ca.vertcat(0, 0, T)) / mass

    # Rotational dynamics
    f_ang = ang_vel
    f_ang_vel = ca.mtimes(I_inv, (tau_beta - ca.cross(ang_vel, ca.mtimes(I, ang_vel))))

    # Combine the dynamics
    dx = ca.vertcat(f_pos, f_ang, f_vel, f_ang_vel)
    return dx

# Rotation matrix function
def rotation_matrix(angles):
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

# MOJE POKUSAVAS
# Closed-loop simulation
xHistory = np.zeros((N_sim+1, n_states))
uHistory = np.zeros((N_sim, n_controls))
xHistory[0, :] = x0  # Initial state

for k in range(N_sim):
    # Set references for previewing
    yref = np.array([QuadrotorReferenceTrajectory(k * Ts) for _ in range(N+1)]).T

    xk = xHistory[k, :]
    # Compute the control moves with reference previewing (using NMPC)
    #x_opt, u_opt = solve_nmpc(xk, yref)
    x_opt, u_opt = solve_nmpc(x0, xk, yref, mv_target)

    #print("Optimal x_opt full solution:", x_opt)
    #print("Optimal u_opt full solution:", u_opt)

    if x_opt is not None and u_opt is not None:
        # Use the first predicted state as the next state
        xk1 = x_opt[:,0]
        xHistory[k+1, :] = xk1

        # Store the first control input
        uk = u_opt[:, 0]
        uHistory[k, :] = uk

        # Apply the control input to the system dynamics to get the next state
        xk1 = dynamics(xk1, uk).full().flatten()
        # Print the optimal control inputs
        print(f"Time: {k * Ts:.2f}, Optimal Control: {uk}")
        # Print the optimal control inputs
        print(f"Time: {k * Ts:.2f}, Optimal Position: {x_opt[:, 0]}")

# Plotting (adjust to match MATLAB plots)
time = np.arange(0, Duration + Ts, Ts)

yref_tot = np.array([QuadrotorReferenceTrajectory(np.array([t_step])) for t_step in time]).squeeze()

# Create subplots for states and control inputs
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Quadrotor States and Control Inputs', fontsize=16)


# Plot the states
for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].plot(time, xHistory[:, i], label='Actual State')
    axs[row, col].plot(time, yref_tot[:, i], label='Reference', linestyle='--')
    axs[row, col].set_xlabel('Time')
    axs[row, col].set_ylabel(['x', 'y', 'z', 'phi', 'theta', 'psi'][i])
    axs[row, col].set_title(['Position x', 'Position y', 'Position z', 'Angle phi', 'Angle theta', 'Angle psi'][i])
    axs[row, col].legend()
    axs[row, col].grid(True)

# Plot the control inputs
fig2, axs2 = plt.subplots(2, 2, figsize=(14, 8))

for i in range(n_controls):
    row = i // 2
    col = i % 2
    axs2[row, col].plot(time[:-1], uHistory[:, i], label=f'Control {i+1}')
    axs2[row, col].set_xlabel('Time')
    axs2[row, col].set_ylabel(f'Control Input {i+1}')
    axs2[row, col].set_title(f'Control Input {i+1} Over Time')
    axs2[row, col].legend()
    axs2[row, col].grid(True)

# Create a 3D plot for x, y, z positions
fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot(xHistory[:, 0], xHistory[:, 1], xHistory[:, 2], label='Actual Position')
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


# Assuming obj is your objective function and g is your constraints vector
#mpc_function = ca.Function('mpc_solver', [X, U, P], [obj, ca.vertcat(*g)])

# Generate C code
#mpc_function.generate('mpc_solver.c')