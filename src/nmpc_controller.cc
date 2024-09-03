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
    b = 0.2;  // Drag constant

    I_xx = (1.0 / 12.0) * mass * (0.15 * 0.15 + 0.15 * 0.15);      // (1.0 / 12.0) * mass * (height * height + depth * depth)
    I_yy = I_xx;      // Inertia around y-axis
    I_zz = 2*I_xx;      // Inertia around z-axis

    roll_time_constant = 0.253;
    pitch_time_constant = 0.267;
    roll_gain = 1.101;
    pitch_gain = 1.097;
    K_yaw = 1.8;

#if 0
    // All your variable declarations here
    dt = 0.08; // Time step
    b_thrust = 0.2,  b_roll = 0.3, b_pitch = 0.1, b_yaw = 0.2;

    // Example weights for Q matrix
    q_vx = 1.5, q_vy = 1.5, q_vz = 1.5;
    q_phi = 1.0, q_theta = 1.0, q_psi = 1.0, q_p = 0, q_q = 0, q_r = 0;

    // Example weights for R matrix
    r_T = 0.1, r_phi = 0.2, r_theta = 0.1, r_psi = 0.1;

    // Example weights for Q_final matrix
    q_vx_fin = 2.5, q_vy_fin = 2.5, q_vz_fin = 2.5;
    q_phi_fin = 1.0, q_theta_fin = 1.0, q_psi_fin =1.0, q_p_fin = 0, q_q_fin = 0, q_r_fin = 0;
#endif

    ROS_INFO("Pre inicijalizacije");

    initializeMPC();
    ROS_INFO("MPC Controller Initialization Complete");
    return true;
}
     
DM MpcController::computeMPCControl(double current_vx, double current_vy, double current_vz, double current_roll, double current_pitch, double current_yaw, double ang_vel_roll, double ang_vel_pitch, double ang_vel_yaw, double target_vx, double target_vy, double target_vz, double target_roll, double target_pitch, double target_yaw, double target_ang_vel_roll, double target_ang_vel_pitch, double target_ang_vel_yaw) {

  ROS_ERROR("computeMPCControl ( v - rpy - vel rpy) - %f %f %f - %f %f %f - %f %f %f",
            current_vx, current_vy, current_vz,
            current_roll, current_pitch, current_yaw,
            ang_vel_roll, ang_vel_pitch, ang_vel_yaw);
            
  ROS_ERROR("computeMPCControl ( vtarget - rpy - vrpy) - %f %f %f - %f %f %f - %f %f %f",
            target_vx, target_vy, target_vz,
            target_roll, target_pitch, target_yaw,
            target_ang_vel_roll, target_ang_vel_pitch, target_ang_vel_yaw);
  
    
    ROS_ERROR("BAG 0");
    // Initialize current state and target state as CasADi SX
    DM x0 = DM::vertcat({current_roll, current_pitch, current_yaw, 
                         current_vx, current_vy, current_vz, 
                         ang_vel_roll, ang_vel_pitch, ang_vel_yaw});

    DM x_ref = DM::vertcat({target_roll, target_pitch, target_yaw, 
                            target_roll, target_pitch, target_yaw, 
                            target_ang_vel_roll, target_ang_vel_pitch, target_ang_vel_yaw});   
    ROS_ERROR("BAG bravo");
    DM x_ref_full = DM::repmat(x_ref, 1, N + 1);
    x_ref_full = DM::reshape(x_ref_full, x_ref_full.size1() * x_ref_full.size2(), 1);

    // Define control input limits as CasADi SX vectors
    DM u_max = DM::vertcat({max_thrust, max_roll, max_pitch, max_yaw});
    DM u_min = DM::vertcat({min_thrust, min_roll, min_pitch, min_yaw});

    // Set lower and upper bounds for decision variables (control inputs and states)
    std::vector<double> lbx_vec, ubx_vec;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n_controls; ++j) {
            lbx_vec.push_back(u_min(j).scalar()); // Lower bound for control inputs
            ubx_vec.push_back(u_max(j).scalar()); // Upper bound for control inputs
        }
    }
    // Add bounds for states
    for (int i = 0; i < n_states * (N + 1); ++i) {
        lbx_vec.push_back(-20); // Lower bound for states lbx_vec.push_back(-casadi::inf); 
        ubx_vec.push_back(20);  // Upper bound for states  ubx_vec.push_back(casadi::inf); 
    }

    DM lbx = DM(lbx_vec);
    DM ubx = DM(ubx_vec);
    
    // Set lower and upper bounds for constraints (only dynamic constraints)
    DM lbg = DM::zeros(n_states * N); // Dynamics are equality constraints
    DM ubg = DM::zeros(n_states * N);
    
    ROS_ERROR("BAG 1");

    DM u_last = mv_target; 
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
    std::cout << "Optimal full u_opt: " << u_opt_full << std::endl;
    u_opt_full = DM::reshape(u_opt_full, n_controls, N);

    // Extract the state variables
    DM x_opt_full = full_solution(Slice(n_controls * N, n_controls * N + n_states * (N + 1)));
    // Reshape x_opt if necessary
    x_opt_full = DM::reshape(x_opt_full, n_states, N + 1);

    DM x_opt = x_opt_full(Slice(), 0);  // All rows, first column
    DM u_opt = u_opt_full(Slice(), 0);  // All rows, first column

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
    return u_opt; 
 }

// Define the drone's dynamics as a nonlinear function
SX MpcController::drone_dynamics(const SX& x, const SX& u) {
    // Extract states
    SX angles = x(Slice(0, 3));  // Orientation: phi, theta, psi
    SX vel = x(Slice(3, 6));  // Linear velocities: xdot, ydot, zdot
    SX ang_vel = x(Slice(6, 9));  // Angular velocities: phidot, thetadot, psidot

    // Control inputs
    SX thrust = u(0);      // Total thrust
    SX roll_ref = u(1);    // Roll angle in radians
    SX pitch_ref = u(2);   // Pitch angle in radians
    SX yaw_rate_ref = u(3);  // Yaw rate in radians per second

    // Rotation matrix from body frame to inertial frame
    SX R = rotation_matrix(angles);

    SX I = diag(SX::vertcat({I_xx, I_yy, I_zz}));

    // Translational dynamics
    SX drag_acc = - b * vel;

    // Equations for COM configuration
    SX f_com = -gravity * SX::vertcat({0, 0, 1}) + mtimes(R, SX::vertcat({0, 0, thrust})) / mass + drag_acc;

    //  Rotational dynamics
    SX droll = (1 / roll_gain) * (roll_ref - angles(0));
    SX dpitch = (1 / pitch_gain) * (pitch_ref - angles(1));
    SX dyaw = K_yaw*(yaw_rate_ref - ang_vel(2));
    // Combine the dynamics
    SX f = SX::vertcat({vel, droll, dpitch, dyaw, f_com});

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
    Q = SX::zeros(9, 9);
    // Assign values to the diagonal of Q
    Q(0,0) = q_phi; Q(1,1) = q_theta; Q(2,2) = q_psi;
    Q(3,3) = q_vx; Q(4,4) = q_vy; Q(5,5) = q_vz;
    Q(6,6) = q_p; Q(7,7) = q_q; Q(8,8) = q_r;

    R = SX::zeros(4, 4);
    // Assign values to the diagonal of R
    R(0,0) = r_T; R(1,1) = r_phi; R(2,2) = r_theta; R(3,3) = r_psi;

    Q_final = SX::zeros(9, 9);
    // Assign values to the diagonal of Q_final
    Q_final(0,0) = q_phi_fin; Q_final(1,1) = q_theta_fin; Q_final(2,2) = q_psi_fin;
    Q_final(3,3) = q_vx_fin; Q_final(4,4) = q_vy_fin; Q_final(5,5) = q_vz_fin;
    Q_final(6,6) = q_p_fin; Q_final(7,7) = q_q_fin; Q_final(8,8) = q_r_fin;

    //std::cout << "Matrix Q:\n" << Q << std::endl;
    //std::cout << "Matrix R:\n" << R << std::endl;
    //std::cout << "Matrix R:\n" << Q_final << std::endl;

    n_states = 9;
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
        SX st_sx = SX(st);
        SX P_sx = SX(P(Slice(n_states*k, n_states*(k+1))));
        SX Q_sx = SX(Q);
        obj += mtimes(mtimes(transpose(st_sx - P_sx), Q_sx), st_sx - P_sx);

        // Control cost, penalizing deviation from mv_target
        //SX mv_target_sx = SX::vertcat({mv_target[0], mv_target[1], mv_target[2], mv_target[3]});
        SX mv_target_sx = SX::vertcat({mv_target(0), mv_target(1), mv_target(2), mv_target(3)});
        obj += mtimes(mtimes(transpose(con - mv_target_sx), SX(R)), con - mv_target_sx);

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
    SX diff_X_P = SX(X(Slice(), N)) - SX(P(Slice(n_states*N, n_states*(N+1))));
    SX transposed_diff = transpose(diff_X_P);
    obj += mtimes(mtimes(transposed_diff, SX(Q_final)), diff_X_P);

    // Create an NLP solver
    SXDict nlp = {{"x", vertcat(reshape(U, -1, 1), reshape(X, -1, 1))},
                {"f", obj},
                {"g", vertcat(g)},
                {"p", P}};
    Dict opts = {{"ipopt.print_level", 3}, {"print_time", 1}, 
                {"ipopt.max_iter", 1000}, {"ipopt.tol", 1e-6}, 
                {"ipopt.acceptable_tol", 1e-8}, {"ipopt.linear_solver", "mumps"}}; //mumps

    // Create the solver
    nmpc_solver = nlpsol("nmpc_solver", "ipopt", nlp, opts);
} 

// Destructor
MpcController::~MpcController() {
    // Reset matrices and vectors
    // Add any other cleanup code here if needed
}

void MpcController::notify_position(double x, double y, double z) {
   current_x = x;
   current_y = y;
   current_z = z;
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

void MpcController::control(double time_seconds) {

    DM control_signals;
    ROS_INFO("Usli smo u control");
    control_signals = computeMPCControl(current_vx, current_vy, current_vz, current_roll, current_pitch, current_yaw, ang_vel_roll, ang_vel_pitch, ang_vel_yaw, target_vx, target_vy, target_vz, 
    target_roll, target_pitch, target_yaw, target_ang_vel_roll, target_ang_vel_pitch, target_ang_vel_yaw);

    // Extract the control inputs from the u_opt vector
    double optimal_thrust = control_signals(0).scalar();
    double optimal_roll = control_signals(1).scalar();
    double optimal_pitch = control_signals(2).scalar();
    double optimal_yaw = control_signals(3).scalar();

    set_thrust(optimal_thrust);
    set_roll(optimal_roll);
    set_pitch(optimal_pitch);
    set_yaw_rate(optimal_yaw);
    
}

void MpcController::set_target_position(double x, double y, double z) {
    target_x = x;
    target_y = y;
    target_z = z;
    ROS_ERROR("set_target_position: %f %f %f", x, y, z);
}

void MpcController::set_target_yaw(double yaw) {   // you can ignore this or ut it to 0
  ROS_ERROR("set_target_yaw: %f", yaw);
  target_yaw = yaw*M_PI/180.0;
}

void MpcController::set_time_to_target(double time_to_target) {

}

void MpcController::set_non_existant() {  
    const double desired_time_to_target = 10.0; // 1 second to reach the target

    target_vx = (target_x - current_x) / desired_time_to_target; 
    target_vy = (target_y - current_y) / desired_time_to_target;
    target_vz = (target_z - current_z) / desired_time_to_target;

    target_roll = calculateTargetRoll(desired_time_to_target); 
    target_pitch = calculateTargetPitch(desired_time_to_target); 

    target_ang_vel_roll = (target_roll - current_roll) / desired_time_to_target;
    target_ang_vel_pitch = (target_pitch - current_pitch) / desired_time_to_target;
    target_ang_vel_yaw = (target_yaw - current_yaw) / desired_time_to_target;
}

// BECAUSE WE WANT THE SPEED TO BE 1m/s

double MpcController::calculateTargetRoll(double desired_time_to_target) {
    double desired_vy = (target_y - current_y) / desired_time_to_target;
    double k_roll = 0.1; // Gain for roll control, tune based on testing    
    return k_roll * (desired_vy - current_vy); // Adjust roll to achieve target velocity in y
}

double MpcController::calculateTargetPitch(double desired_time_to_target) {
    double desired_vx = (target_x - current_x) / desired_time_to_target;
    double k_pitch = 0.1; // Gain for pitch control, tune based on testing
    return k_pitch * (desired_vx - current_vx); // Adjust pitch to achieve target velocity in x
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
    double error_z = target_z - current_z;
    return error_z;
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


