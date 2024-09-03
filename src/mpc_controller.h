#ifndef MPC_CONTROLLER_H
#define MPC_CONTROLLER_H

#include "controller.h"
//#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <vector>

using namespace casadi;

class MpcController : public virtual Controller {
private:
  double current_x;
  double current_y;
  double current_z;
  double current_vx;
  double current_vy;
  double current_vz;
  double current_roll;
  double current_pitch;  
  double current_yaw;  
  double ang_vel_roll; 
  double ang_vel_pitch; 
  double ang_vel_yaw; 
  double target_x;
  double target_y;
  double target_z;
  double target_yaw;
  double qx, qy, qz, qw;
  double target_vx, target_vy, target_vz;
  double target_roll, target_pitch;
  double target_ang_vel_roll, target_ang_vel_pitch, target_ang_vel_yaw;

  // MPC STUFF THAT NEEDS TO BE TUNED
  double b_thrust, b_roll, b_pitch, b_yaw;
  double I_xx, I_yy, I_zz, mg;      

  // Casadi initialization
  SX A, B, Q, R, Q_final,D, T_hat, I_inv, R_mv_target, R_mv_rate;
  int n_states;
  int n_controls;
  int N; // Horizon length
  Function nmpc_solver; // CasADi Function for MPC solver

  DM x_opt_full;
  std::vector<double> x0;
  std::vector<double> xref;
  DM u_last;
  double current_time;
  SX obj;
  std::vector<SX> g;
  
public:
  MpcController();
  virtual ~MpcController();

  void initializeMPC();

 // DM computeMPCControl(double current_x, double current_y, double current_z, double current_vx, double current_vy, double current_vz, double current_roll, double current_pitch, double current_yaw, double ang_vel_roll, double ang_vel_pitch, double ang_vel_yaw, double
 // target_x, double target_y, double target_z, double target_vx, double target_vy, double target_vz, double target_roll, double target_pitch, double target_yaw, double target_ang_vel_roll, double target_ang_vel_pitch, double target_ang_vel_yaw);
  DM computeMPCControl(double current_vx, double current_vy, double current_vz, double current_roll, double current_pitch, double current_yaw, double ang_vel_roll, double ang_vel_pitch, double ang_vel_yaw, double target_vx, double target_vy, double target_vz, double target_roll, double target_pitch, double target_yaw, double target_ang_vel_roll, double target_ang_vel_pitch, double target_ang_vel_yaw);

  SX drone_dynamics(const SX& x, const SX& u);
  SX rotation_matrix(const SX& angles);
  SX transformation_matrix(const SX& angles);

  // CLOSE LOOP SIMULATION
  DM generateReferenceTrajectory(double current_time, double Ts, int N);
  DM integrateDynamics(const DM& x, const DM& u, double dt);
  DM drone_dynamics_numerical(const DM& x, const DM& u);
  DM rotation_matrix_numerical(const DM& angles);
  DM computeMPCControl2(DM& x_current, DM& yref_transposed, DM& mv_target, DM& u_last);
  
  // ------------------------------------------------------------------------------------------

  //void populate_reference_trajectory(double* x_ref_data, const Eigen::VectorXd& x_ref);
  void notify_position(double x, double y, double z);
  void notify_velocity(double vx, double vy, double vz);
  void notify_attitude(double x, double y, double z, double w);
  void notify_angles(double roll, double pitch, double yaw);
  void notify_angle_rates(double roll_rate, double pitch_rate, double yaw_rate);
  void set_non_existant();
  double calculateTargetRoll(double desired_time_to_target);
  double calculateTargetPitch(double desired_time_to_target);
  void control(double time_seconds);
  void set_target_position(double x, double y, double z);
  void set_target_velocity(double vx, double vy, double vz);
  void set_target_yaw(double yaw);
  void set_time_to_target(double time_to_target);
  double get_err_pitch();
  double get_err_roll();
  double get_err_z();
  bool initialize();
  std::vector<double> get_positions();
  std::vector<double> get_state();
  std::vector<double> get_ref_state();
};

#endif
