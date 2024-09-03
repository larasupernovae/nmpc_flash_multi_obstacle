#include "mpc_controller.h"
#include <iostream> 
#include <sstream>
#include <math.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float32.h>
#include <Eigen/Dense>
#include <vector>
#include <ros/ros.h>
#include "tf/transform_datatypes.h"
#include <limits>

#include <casadi/casadi.hpp>
using namespace casadi;

// Constructor
MpcController::MpcController() {
    
}

DM mv_target = DM::vertcat({37.011, 0.0, 0.0, 0.0});

bool MpcController::initialize () {

    // Initialize member variables
    max_thrust = 80;
    max_roll = 40.0 * M_PI / 180.0;
    max_pitch = 40.0 * M_PI / 180.0;
    //max_pitch_rate = 5.0/6.0 * M_PI;   // According to doc
    //max_roll_rate = 5.0/6.0 * M_PI;   // According to doc
    max_yaw = 5.0/6.0 * M_PI;  // According to doc
    min_thrust = 20;   // 37 it hovers...0-80
    min_roll = -40.0 * M_PI / 180.0;
    min_pitch = -40.0 * M_PI / 180.0;
    //min_pitch_rate = -max_pitch_rate;
    //min_roll_rate = -max_roll_rate;
    min_yaw = -max_yaw;

    // Drone parameters
    mass = 2.895;
    gravity = 9.81;
    mg = mass * gravity;  // mass times gravity

    k = 1.0; // Lift constant
    l = 0.25; // Distance between rotor and COM
    b = 0.1;  // Drag constant

    I_xx = (1.0 / 12.0) * mass * (0.15 * 0.15 + 0.15 * 0.15);      // (1.0 / 12.0) * mass * (height * height + depth * depth)
    I_yy = I_xx;      // Inertia around y-axis
    I_zz = 2*I_xx;      // Inertia around z-axis

    // All your variable declarations here
    dt = 0.08; // Time step

    // Example weights for Q matrix
    q_x = 1.0, q_y = 1.0, q_z = 1.0;          // Increased positional weights
    q_vx = 0.1, q_vy = 0.1, q_vz = 0.1;       // Kept the same for velocity
    q_phi = 0.6, q_theta = 0.6, q_psi = 0.1;  // Orientation may not be as critical
    q_r = 1.0;                                // Angular rates are kept the same

    // Modified weights for R matrix
    r_T = 0.03;                                // Reduced weight for thrust
    r_phi = 0.1, r_theta = 0.1, r_psi = 0.1;   // Kept the same for other controls

    // Example weights for Q_final matrix
    q_x_fin = 1.0, q_y_fin = 1.0, q_z_fin = 1.0;     // Increased final positional weights
    q_vx_fin = 0.1, q_vy_fin = 0.1, q_vz_fin = 0.1;
    q_phi_fin = 0.6, q_theta_fin = 0.6, q_psi_fin = 0.1;
    q_r_fin = 1.0;

    roll_time_constant = 0.253;
    pitch_time_constant = 0.267;
    roll_gain = 1.101;
    pitch_gain = 1.097;
    K_yaw = 1.8;

    ROS_INFO("Pre inicijalizacije");

    initializeMPC();
    ROS_INFO("MPC Controller Initialization Complete");
    return true;
}
     
//std::pair<DM, DM>
    DM MpcController::computeMPCControl2(DM& x_current, DM& yref_transposed, DM& mv_target, DM& u_last) {

    //std::cout << "x_current: " << x_current << std::endl;
    //std::cout << "yref_transposed: " << yref_transposed << std::endl;
    //std::cout << "mv_target: " << mv_target << std::endl;
    //std::cout << "u_last: " << u_last << std::endl;

    ROS_ERROR("BAG bravo");
    DM x_ref_full = yref_transposed;
    DM x0 = x_current;
    //DM x_ref_full = DM::repmat(x_ref, 1, N + 1);
    x_ref_full = DM::reshape(x_ref_full, x_ref_full.size1() * x_ref_full.size2(), 1);

    // Define control input limits as CasADi SX vectors
    DM u_max = DM::vertcat({max_thrust, max_roll, max_pitch, max_yaw});
    DM u_min = DM::vertcat({min_thrust, min_roll, min_pitch, min_yaw});

    // Initialize DM matrices for lbx and ubx
    DM lbx = DM::zeros(N * n_controls + n_states * (N + 1));
    DM ubx = DM::zeros(N * n_controls + n_states * (N + 1));

    // Set bounds for control inputs
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n_controls; ++j) {
            lbx(i * n_controls + j) = u_min(j); // Lower bound for control inputs
            ubx(i * n_controls + j) = u_max(j); // Upper bound for control inputs
        }
    }

    // Get the total number of elements in the lbx matrix
    int lbx_total_elements = lbx.size1() * lbx.size2();

    // Set bounds for states
    for (int i = N * n_controls; i < lbx_total_elements; ++i) {
    lbx(i) = -std::numeric_limits<double>::infinity(); // Lower bound for states
    ubx(i) = std::numeric_limits<double>::infinity();  // Upper bound for states
}

    // Set lower and upper bounds for constraints (only dynamic constraints)
    DM lbg = DM::zeros(n_states * N); // Dynamics are equality constraints
    DM ubg = DM::zeros(n_states * N);

    //std::cout << "lbx: " << lbx << std::endl;
    //std::cout << "ubx: " << ubx << std::endl;
    //std::cout << "lbg: " << lbg << std::endl;
    //std::cout << "ubg: " << ubg << std::endl;
    
    ROS_ERROR("BAG 1");

    //u_last = mv_target; 
    DM neutral_control_input = DM::vertcat({0, 0, 0, 0}); // Define neutral control input as DM

    // Define a neutral initial guess for control inputs
    DM u0_repeated;
    if (!u_last.is_empty()) {
        u0_repeated = DM::repmat(u_last, N, 1); // Repeat u_last N times
    } else {
        u0_repeated = DM::repmat(neutral_control_input, N, 1); // Repeat neutral control input N times
    }


    ROS_ERROR("P1");    
    // Convert u0_repeated to DM
    DM u0_repeated_DM = DM(u0_repeated);

    // Concatenate initial guesses for control inputs and state variables
    DM initial_guess = DM::vertcat({u0_repeated_DM, DM::repmat(x0, N + 1, 1)});

    //std::cout << "initial_guess: " << initial_guess << std::endl;
    //std::cout << "x_ref_full: " << x_ref_full << std::endl;

    // Prepare the arguments for the solver
    std::map<std::string, DM> args;
    args["x0"] = initial_guess; // Initial guess
    args["p"] = x_ref_full; // Parameters (initial state and reference trajectory)
    args["lbx"] = lbx; // Lower bound for decision variables
    args["ubx"] = ubx; // Upper bound for decision variables
    args["lbg"] = lbg; // Lower bound for constraints
    args["ubg"] = ubg; // Upper bound for constraints

    // Print the arguments before passing them to the solver
    //for (const auto& arg : args) {
    //    std::cout << "Key: " << arg.first << ", Value: " << arg.second << std::endl;
    //}

    // Call the solver
    DMDict res = nmpc_solver(args);

    ROS_ERROR("P2");

    // Extract the full solution
    DM full_solution = res["x"];

    // Extract the optimal control inputs
    DM u_opt_full = full_solution(Slice(0, n_controls * N));
    // Reshape u_opt if necessary
    //std::cout << "Optimal full u_opt: " << u_opt_full << std::endl;
    u_opt_full = DM::reshape(u_opt_full, n_controls, N);
    //std::cout << "u_opt_full: " << u_opt_full << std::endl;
    
    // Extract the state variables
    DM x_opt_full = full_solution(Slice(n_controls * N, n_controls * N + n_states * (N + 1)));
    //std::cout << "x_opt_full: " << x_opt_full << std::endl;
    // Reshape x_opt if necessary
    x_opt_full = DM::reshape(x_opt_full, n_states, N + 1);
    //std::cout << "x_opt_full reshaped: " << x_opt_full << std::endl;
    
    DM x_opt = x_opt_full; //(Slice(), 0);  // All rows, first column
    DM u_opt = u_opt_full; //(Slice(), 0);  // All rows, first column

    //std::cout << "x_opt: " << x_opt << std::endl;

    // u_opt_full = DM::reshape(u_opt_full, n_controls, N);
    //std::cout << "u_opt" << repr(u_opt) << std::endl;
    ROS_ERROR("P3");
    ROS_ERROR("N: %d", N);
    ROS_ERROR("n_states: %d", n_states);
    //int size = sizeof(res["x"])
    //ROS_ERROR("size res["x"]: %d", size);

                  // std::cout << repr(res["x"]);
    
    // Extract control inputs and states from the full solution
    ///std::vector<double> u_opt = std::vector<double>(u_opt_full.nonzeros().begin(), u_opt_full.nonzeros().end());
    ROS_ERROR("P4");
    ///    std::vector<double> first = std::vector<double>(res["x"](Slice(n_controls*N, -1));
    ///    std::vector<double> x_opt = std::vector<double>(res["x"](Slice(0, n_states * (N + 1))).nonzeros().begin(), res["x"](Slice(0, n_states * (N + 1))).nonzeros().end());

    ROS_ERROR("P5");        
    // Extract the control inputs for the first time step
    //std::cout << "Optimal control inputs u_opt_first:: " << u_opt_first << std::endl;
    std::cout << "Optimal control inputs u_opt: " << u_opt << std::endl;
    std::cout << "Optimal states x_opt: " << x_opt << std::endl;
    //return {x_opt, u_opt};
    return u_opt;
 }

// Define the drone's dynamics as a nonlinear function
SX MpcController::drone_dynamics(const SX& x, const SX& u) {
    // Extract states
    SX pos = x(Slice(0, 3));  // States: x, y, z
    SX angles = x(Slice(3, 6));  // Orientation: phi, theta, psi
    SX vel = x(Slice(6, 9));  // Linear velocities: xdot, ydot, zdot
    SX ang_vel = x(9);  // Angular velocity: yaw_rate

    // Control inputs
    SX thrust = u(0);      // Total thrust
    SX roll_ref = u(1);    // Roll angle in radians
    SX pitch_ref = u(2);   // Pitch angle in radians
    SX yaw_rate_ref = u(3);  // Yaw rate in radians per second

    // Rotation matrix from body frame to inertial frame
    SX R = rotation_matrix(angles);

    // Translational dynamics
    // Drag coefficients (example values, adjust as needed)
    b_x = 0.1;  // Drag coefficient in x direction
    b_y = 0.1;  // Drag coefficient in y direction
    b_z = 0.1;  // Drag coefficient in z direction

    // Drag acceleration in body frame
    SX drag_acc_intertial = -SX::vertcat({b_x * vel(0), b_y * vel(1), b_z * vel(2)});

    // Gravity vector in inertial frame
    SX gravity_vector = SX::vertcat({0, 0, -gravity});

    // Thrust vector in body frame
    SX thrust_vector = SX::vertcat({0, 0, thrust});

    // Equations for COM configuration
    // Translational dynamics
    SX translational_dynamics = gravity_vector + mtimes(R, thrust_vector) / mass + mtimes(R, drag_acc_intertial);

    // Rotational dynamics
    SX droll = (1 / roll_time_constant) * (roll_gain * (roll_ref - angles(0)));
    SX dpitch = (1 / pitch_time_constant) * (pitch_gain * (pitch_ref - angles(1)));
      
    SX dyaw = angles(2);
   
    SX ddyaw = - K_yaw * (yaw_rate_ref - ang_vel);

    SX rotational_dynamics = SX::vertcat({droll, dpitch, dyaw});
    // Combine the dynamics
    SX f = SX::vertcat({vel, rotational_dynamics, translational_dynamics, ddyaw});

    return f;
}

// Rotation matrix function
SX MpcController::rotation_matrix(const SX& angles) {
    SX phi = angles(0), theta = angles(1), psi = angles(2);

    // Rotation matrices for Rz, Ry, Rx
    SX Rz = SX::vertcat({
        SX::horzcat({cos(psi), -sin(psi), 0}),
        SX::horzcat({sin(psi), cos(psi), 0}),
        SX::horzcat({0, 0, 1})
    });

    SX Ry = SX::vertcat({
        SX::horzcat({cos(theta), 0, sin(theta)}),
        SX::horzcat({0, 1, 0}),
        SX::horzcat({-sin(theta), 0, cos(theta)})
    });

    SX Rx = SX::vertcat({
        SX::horzcat({1, 0, 0}),
        SX::horzcat({0, cos(phi), -sin(phi)}),
        SX::horzcat({0, sin(phi), cos(phi)})
    });

    SX R_dyn = mtimes(Rz, mtimes(Ry, Rx));

    return R_dyn;
}

void MpcController::initializeMPC() {
    // All your initialization code here
    D = SX::zeros(3, 3);
    T_hat = SX::zeros(3, 3);
    I_inv = SX::zeros(3, 3);

    ROS_INFO("Initialized matrices with appropriate sizes.");
    // Example values for the drag matrix D
    D = SX({{0.1, 0, 0},
            {0, 0.1, 0},
            {0, 0, 0.1}});

    // Example values for the thrust transformation matrix T_hat
    // Assuming the thrust is aligned with the z-axis of the drone
    T_hat = SX({{0, mg, 0},
                {-mg, 0, 0},
                {0, 0, 0}});

    // Example values for the inverse of the inertia matrix I_inv
    // Replace these values with your drone's specific inertia matrix values
    I_inv = SX({{I_xx, 0, 0},
                {0, I_yy, 0},
                {0, 0, I_zz}});

   // std::cout << "Matrix A:\n" << A << std::endl;
   // std::cout << "Matrix B:\n" << B << std::endl;

    // Initialize matrices using CasADi SX
    Q = SX::zeros(10, 10);
    // Assign values to the diagonal of Q
    Q(0,0) = q_x; Q(1,1) = q_y; Q(2,2) = q_z;
    Q(3,3) = q_phi; Q(4,4) = q_theta; Q(5,5) = q_psi;
    Q(6,6) = q_vx; Q(7,7) = q_vy; Q(8,8) = q_vz;
    Q(9,9) = q_r;

    R = SX::zeros(4, 4);
    // Assign values to the diagonal of R
    R(0,0) = r_T; R(1,1) = r_phi; R(2,2) = r_theta; R(3,3) = r_psi;

    Q_final = SX::zeros(10, 10);
    // Assign values to the diagonal of Q_final
    Q_final(0,0) = q_x; Q_final(1,1) = q_y; Q_final(2,2) = q_z;
    Q_final(3,3) = q_phi; Q_final(4,4) = q_theta; Q_final(5,5) = q_psi;
    Q_final(6,6) = q_vx; Q_final(7,7) = q_vy; Q_final(8,8) = q_vz;
    Q_final(9,9) = q_r;
    
    R_mv_target = SX::zeros(4, 4);
    R_mv_rate = SX::zeros(4, 4);
    // Add new weight matrices for MV targets and MV rate of change
    R_mv_target(0,0) = 0.1; R_mv_target(1,1) = 0.1; R_mv_target(2,2) = 0.1; R_mv_target(3,3) = 0.1;  // Tuning weight for MV target  
    R_mv_rate(0,0) = 0.1; R_mv_rate(1,1) = 0.1; R_mv_rate(2,2) = 0.1; R_mv_rate(3,3) = 0.1;  // Tuning weight for MV rate of change

    //std::cout << "Matrix Q:\n" << Q << std::endl;
    //std::cout << "Matrix R:\n" << R << std::endl;
    //std::cout << "Matrix R:\n" << Q_final << std::endl;

    n_states = 10;
    n_controls = 4;
    N = 24;  // Horizon length
    std::cout << "Time horizon: " << N << std::endl;

    // Define the state and control trajectory symbolsr
    SX X = SX::sym("X", n_states, N+1);  // State trajectory
    SX U = SX::sym("U", n_controls, N);  // Control trajectory
    SX P = SX::sym("P", n_states * (N + 1));  // Parameters

    // Objective function and constraints
    SX obj; // Objective function
    std::vector<SX> g; // Constraints vector
    obj = 0; // Initialize objective function

    ROS_ERROR("Pre for petlje novo");

    // Nominal control that keeps the quadrotor floating - reference control
    // std::vector<double> mv_target = {37.011, 4.9*M_PI/180.0, 4.9*M_PI/180.0, 4.9*M_PI/180.0};

    // Add control input constraints and system dynamics constraints
    for (int k = 0; k < N; ++k) {
        SX con = U(Slice(), k);
        SX st = X(Slice(), k);

        // State cost
        //SX st_sx = SX(st);
        //SX P_sx = SX(P(Slice(n_states*k, n_states*(k+1))));
        //SX Q_sx = SX(Q);
        //obj += mtimes(mtimes(transpose(st_sx - P_sx), Q_sx), st_sx - P_sx);
        obj += mtimes(mtimes(transpose(st - P(Slice(n_states*k, n_states*(k+1)))), Q), st - P(Slice(n_states*k, n_states*(k+1))));

        // Control cost, penalizing deviation from mv_target
        obj += mtimes(mtimes(transpose(con), SX(R)), con);
        
        SX mv_target_sx = SX::vertcat({mv_target(0), mv_target(1), mv_target(2), mv_target(3)});
        obj += mtimes(mtimes(transpose(con - mv_target_sx), SX(R_mv_target)), con - mv_target_sx);

        // For the rate of change, compare to the previous control input if not the first step
        if (k > 0) {
            SX con_prev = U(Slice(), k-1);
            obj += mtimes(mtimes(transpose(con - con_prev), SX(R_mv_rate)), con - con_prev);
        }

        // Dynamics constraint
        SX st_next = X(Slice(), k+1);
        SX f = drone_dynamics(st, con); 
        // System dynamics constraint
        SX dynamics_constraint = st_next - (st + dt * f);
        g.push_back(dynamics_constraint);

        // Print the dynamics constraint
        //std::cout << "Dynamics constraint at step " << k << ": " << dynamics_constraint << std::endl;
    }

    // Final state cost
    std::cout << "Size of X: " << X.size() << std::endl;
    std::cout << "Size of P: " << P.size() << std::endl;

    // Final state cost
    //SX diff_X_P = SX(X(Slice(), N)) - SX(P(Slice(n_states*N, n_states*(N+1))));
    //SX transposed_diff = transpose(diff_X_P);
    //obj += mtimes(mtimes(transposed_diff, SX(Q_final)), diff_X_P);
    obj += mtimes(mtimes(transpose(X(Slice(), N) - P(Slice(n_states*N, n_states*(N+1)))), Q_final), X(Slice(), N) - P(Slice(n_states*N, n_states*(N+1))));


    // Create an NLP solver
    SXDict nlp = {{"x", vertcat(reshape(U, -1, 1), reshape(X, -1, 1))},
                {"f", obj},
                {"g", vertcat(g)},
                {"p", P}};

    Dict opts = {{"ipopt.print_level", 0}, {"print_time", 0},   // 3 , 1
                {"ipopt.max_iter", 1000}, {"ipopt.tol", 1e-10}, 
                {"ipopt.acceptable_tol", 1e-12}, {"ipopt.linear_solver", "mumps"}}; //mumps

    // Create the solver

    //std::cout << "create_solver nlp: " << nlp << std::endl;
    //std::cout << "create_solver opts: " << opts << std::endl;
    
    nmpc_solver = nlpsol("nmpc_solver", "ipopt", nlp, opts);

    x0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Replace with your initial state values
    u_last = DM(mv_target);
    current_time = 0.0;
} 

// Destructor
MpcController::~MpcController() {
    // Reset matrices and vectors
    // Add any other cleanup code here if needed
}
/*
DM MpcController::generateReferenceTrajectory(double current_time, double Ts, int N) {
    DM yref = DM::zeros(n_states, N + 1);    
    // Fill yref with your trajectory data

    for (int i = 0; i <= N; ++i) {
      #if 0
        double t = current_time + i * Ts;
        // Calculate the desired trajectory
        double x = 6 * sin(t / 3);
        double y = -6 * sin(t / 3) * cos(t / 3);
        double z = 6 * cos(t / 3);

        // Derivatives of the trajectory (velocity components)
        double xdot = 2 * cos(t / 3); // derivative of x with respect to t
        double ydot = -2 * (sin(t / 3) + cos(t / 3) * cos(t / 3)); // derivative of y with respect to t
        double zdot = -2 * sin(t / 3); // derivative of z with respect to t

#endif
        
        // Initialize other states to zero
        double phi = 0, theta = 0, psi = 0;
        double phidot = 0, thetadot = 0, psidot = 0;
        double xdot = target_vx;
        double ydot = target_vy;
        double zdot = target_vz;
        
        // Assign values to yrefts
        yref(0, i) = phi;
        yref(1, i) = theta;
        yref(2, i) = psi;
        yref(3, i) = xdot;
        yref(4, i) = ydot;
        yref(5, i) = zdot;
        yref(6, i) = phidot;
        yref(7, i) = thetadot;
        yref(8, i) = psidot;
    }

    // Transpose yref to match the expected format
    DM yref_transposed = yref.T(); // Transpose the matrix

    return yref_transposed;
}
*/

DM MpcController::generateReferenceTrajectory(double current_time, double Ts, int N) {
    // Define the initial position and velocity
    double x0 = 0.0, y0 = 0.0, z0 = 1.0;  // Initial position
    double v = 5.0;  // Velocity

    // Preallocate the trajectory DM matrix
    DM xdesired = DM::zeros(10, N+1); // 10 states over N+1 time steps

    for (int k = 0; k <= N; ++k) {
        double t = current_time + k * Ts; // Calculate the time for each step

        // Calculate the desired trajectory
        xdesired(0, k) = t < 10.0 ? x0 + v * t : x0 + v * 10.0;
        xdesired(1, k) = t < 10.0 ? y0 + v/2.0 * t : y0 + v/2.0 * 10.0;
        xdesired(2, k) = z0; // z is constant
        xdesired(6, k) = t < 10.0 ? v : 0.0; // xdot
        xdesired(7, k) = t < 10.0 ? v/2.0 : 0.0; // ydot
        // zdot, phi, theta, psi, and psidot remain zero, as initialized
    }

    return xdesired;
}

DM MpcController::integrateDynamics(const DM& x, const DM& u, double dt) {
    // Your drone_dynamics_numerical function should correctly implement the drone's dynamics
    // Assuming drone_dynamics_numerical is already correctly implemented

    DM x_new = x; // Initialize new state

    // Assuming the dynamics function returns the state derivative, use Euler integration
    DM dx = drone_dynamics_numerical(x, u);
    for (int i = 0; i < x.size1(); ++i) {
        x_new(i) += dt * dx(i); // Simple Euler integration for each state
    }

    return x_new;
}


DM MpcController::drone_dynamics_numerical(const DM& x, const DM& u) {
    // Extract states
    DM pos = x(Slice(0, 3));  // States: x, y, z
    DM angles = x(Slice(3, 6));  // Orientation: phi, theta, psi
    DM vel = x(Slice(6, 9));  // Linear velocities: xdot, ydot, zdot
    DM ang_vel = x(9);  // Angular velocity: yaw_rate

    // Control inputs
    DM thrust = u(0);      // Total thrust
    DM roll_ref = u(1);    // Roll angle in radians
    DM pitch_ref = u(2);   // Pitch angle in radians
    DM yaw_rate_ref = u(3);  // Yaw rate in radians per second

    // Rotation matrix from body frame to inertial frame
    DM R = rotation_matrix_numerical(angles);

    // Translational dynamics
    // Drag coefficients (example values, adjust as needed)
    b_x = 0.1;  // Drag coefficient in x direction
    b_y = 0.1;  // Drag coefficient in y direction
    b_z = 0.1;  // Drag coefficient in z direction

    // Drag acceleration in body frame
    DM drag_acc_intertial = -DM::vertcat({b_x * vel(0), b_y * vel(1), b_z * vel(2)});

    // Gravity vector in inertial frame
    DM gravity_vector = DM::vertcat({0, 0, -gravity});

    // Thrust vector in body frame
    DM thrust_vector = DM::vertcat({0, 0, thrust});

    // Equations for COM configuration
    // Translational dynamics
    DM translational_dynamics = gravity_vector + mtimes(R, thrust_vector) / mass + mtimes(R, drag_acc_intertial);

    // Rotational dynamics
    DM droll = (1 / roll_time_constant) * (roll_gain * (roll_ref - angles(0)));
    DM dpitch = (1 / pitch_time_constant) * (pitch_gain * (pitch_ref - angles(1)));
    DM dyaw = angles(2);

    DM ddyaw = - K_yaw * (yaw_rate_ref - ang_vel);
    // Combine the dynamics

    DM rotational_dynamics = DM::vertcat({droll, dpitch, dyaw});
    // Combine the dynamics
    DM f = DM::vertcat({vel, rotational_dynamics, translational_dynamics, ddyaw});

    return f;
}


DM MpcController::rotation_matrix_numerical(const DM& angles) {
    DM phi = angles(0), theta = angles(1), psi = angles(2);

    // Rotation matrices for Rz, Ry, Rx
    DM Rz = DM::vertcat({
        DM::horzcat({cos(psi), -sin(psi), 0}),
        DM::horzcat({sin(psi), cos(psi), 0}),
        DM::horzcat({0, 0, 1})
    });

    DM Ry = DM::vertcat({
        DM::horzcat({cos(theta), 0, sin(theta)}),
        DM::horzcat({0, 1, 0}),
        DM::horzcat({-sin(theta), 0, cos(theta)})
    });

    DM Rx = DM::vertcat({
        DM::horzcat({1, 0, 0}),
        DM::horzcat({0, cos(phi), -sin(phi)}),
        DM::horzcat({0, sin(phi), cos(phi)})
    });

    DM R_dyn = mtimes(Rz, mtimes(Ry, Rx));

    return R_dyn;
}

void MpcController::control(double time_seconds) {

    ROS_INFO("Usli smo u control");
    
    double Ts = time_seconds; // Time step
    double Duration = 20; // Total duration of simulation
    int N_sim = static_cast<int>(Duration / Ts); // Number of simulation steps

    DM xHistory = DM::zeros(n_states, N_sim + 1);
    DM uHistory = DM::zeros(n_controls, N_sim);

    // Initial state
    DM x_current = DM({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}); // Replace with your initial state values
    // Set initial state in history
    xHistory(Slice(), 0) = x_current;

    DM u_last = DM(mv_target); // Initialize the last optimal control input
    
    for (int k = 0; k < N_sim; ++k) {
        double current_time = k * Ts;

        DM yref = generateReferenceTrajectory(current_time, Ts, N);
        DM yref_transposed = yref.T();

        // Adjusted to receive a pair from computeMPCControl2
        //auto [x_opt, u_opt] = computeMPCControl2(x_current, yref_transposed, mv_target, u_last);
        DM control_signals = computeMPCControl2(x_current, yref_transposed, mv_target, u_last);
        // Now u_opt contains control inputs for the entire horizon, but we only apply the first set
        //DM control_signals = u_opt(Slice(), 0);

        // Apply control inputs (for the first step only, as per usual control logic)
        set_pitch(control_signals(1).scalar());
        set_roll(control_signals(0).scalar());
        set_yaw_rate(control_signals(3).scalar());
        set_thrust(control_signals(2).scalar());

        // Update state based on control inputs, using only the first set of control signals
        DM x_new = integrateDynamics(x_current, control_signals, Ts);

        // Update histories
        xHistory(Slice(), k + 1) = x_new;
        uHistory(Slice(), k) = control_signals;

        // Prepare for next iteration
        x_current = x_new;
        u_last = control_signals; // Update last optimal control to the current applied control
    }
}

void MpcController::notify_position(double x, double y, double z) {

}

void MpcController::notify_velocity(double vx, double vy, double vz) {
   current_vx = vx;
   current_vy = vy;
   current_vz = vz;
}

void MpcController::notify_attitude(double x, double y, double z, double w) {
    qx = x;
    qy = y;
    qz = z;
    qw = w;
    tf::Quaternion quat;
    quat[0] = x;
    quat[1] = y;
    quat[2] = z;
    quat[3] = w;
    double roll, pitch, yaw;
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);

   // current_roll = roll;
   // current_pitch = pitch;
  //  current_yaw = yaw;
}

void MpcController::notify_angles(double roll, double pitch, double yaw) {
  ROS_ERROR("notify_angles: %f %f %f", roll, pitch, yaw);  
    current_roll = roll;
    current_pitch = pitch;
    current_yaw = yaw;
}

void MpcController::notify_angle_rates(double roll_rate, double pitch_rate, double yaw_rate) {
  ROS_ERROR("notify_angle_rates: %f %f %f", roll_rate, pitch_rate, yaw_rate);
  ang_vel_roll = roll_rate; 
  ang_vel_pitch = pitch_rate; 
  ang_vel_yaw = yaw_rate;
}



void MpcController::set_target_position(double x, double y, double z) {
  ROS_ERROR("set_target_position NOOP: %f %f %f", x, y, z);
}

void MpcController::set_target_velocity(double vx, double vy, double vz) {
  target_vx = vx;
  target_vy = vy;
  target_vz = vz;
  ROS_ERROR("set_target_velocity: %f %f %f", vx, vy, vz);
}

void MpcController::set_target_yaw(double yaw) {   // you can ignore this or ut it to 0
  ROS_ERROR("set_target_yaw: %f", yaw);
  target_yaw = yaw*M_PI/180.0;
}

void MpcController::set_time_to_target(double time_to_target) {

}

void MpcController::set_non_existant() {  
    const double desired_time_to_target = 10.0; // 1 second to reach the target

    //target_vx = (target_x - current_x) / desired_time_to_target; 
    //target_vy = (target_y - current_y) / desired_time_to_target;
   // target_vz = (target_z - current_z) / desired_time_to_target;

    target_roll = calculateTargetRoll(desired_time_to_target); 
    target_pitch = calculateTargetPitch(desired_time_to_target); 

    target_ang_vel_roll = (target_roll - current_roll) / desired_time_to_target;
    target_ang_vel_pitch = (target_pitch - current_pitch) / desired_time_to_target;
    target_ang_vel_yaw = (target_yaw - current_yaw) / desired_time_to_target;
}

// BECAUSE WE WANT THE SPEED TO BE 1m/s

double MpcController::calculateTargetRoll(double desired_time_to_target) {
    //double desired_vy = (target_y - current_y) / desired_time_to_target;
    //double k_roll = 0.1; // Gain for roll control, tune based on testing    
    //return k_roll * (desired_vy - current_vy); // Adjust roll to achieve target velocity in y
    return 0.0;
}

double MpcController::calculateTargetPitch(double desired_time_to_target) {
    //double desired_vx = (target_x - current_x) / desired_time_to_target;
    //double k_pitch = 0.1; // Gain for pitch control, tune based on testing
    //return k_pitch * (desired_vx - current_vx); // Adjust pitch to achieve target velocity in x
    return 0.0;
}

double MpcController::get_err_pitch() {
    double error_pitch = target_pitch - current_pitch;
    return error_pitch;
}

double MpcController::get_err_roll() {
    double error_roll = target_roll - current_roll;
    return error_roll;
}

double MpcController::get_err_z() {
   // double error_z = target_z - current_z;
   //return error_z;
    return 0.0;
}


std::vector<double> MpcController::get_positions() {
    std::vector<double> res;

    for (unsigned int i=0; i<(N+1); i++) {
      res.push_back(x_opt_full(n_states*i+0).scalar());
      res.push_back(x_opt_full(n_states*i+1).scalar());
      res.push_back(x_opt_full(n_states*i+2).scalar());
    }
#if 0    
    DM x_opt_slice = x_opt.slice(0, 3);
    DM current_x = x_opt_slice[0];
    DM current_y = x_opt_slice[1];
    DM current_z = x_opt_slice[2];

    // Add the specified target positions
    target_x = 2.0;
    target_y = 0.0;
    target_z = 0.0;

    res.push_back(current_x);
    res.push_back(current_y);
    res.push_back(current_z);
    res.push_back(target_x);
    res.push_back(target_y);
    res.push_back(target_z);

    // Move to the next target positions
    target_x = 2.0;
    target_y = 4.0;
    target_z = 0.0;

    res.push_back(target_x);
    res.push_back(target_y);
    res.push_back(target_z);

    // Move to the final target positions
    target_x = 2.0;
    target_y = 4.0;
    target_z = 3.0;

    res.push_back(target_x);
    res.push_back(target_y);
    res.push_back(target_z);
#endif
    return res;
}

std::vector<double> MpcController::get_state() {
  return x0;
}

std::vector<double> MpcController::get_ref_state() {
  return x0;
}
