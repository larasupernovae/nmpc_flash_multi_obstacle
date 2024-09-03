#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

#include "controller.h"

class PidController : public virtual Controller {
private:
  double err_z;
  double old_err_z;
  double int_err_z;
  double d_err_z;
  double err_yaw, old_err_yaw, int_err_yaw, d_err_yaw;
  double err_pitch, err_roll, old_err_pitch, old_err_roll;
  double target_x;
  double target_y;
  double target_z;
  double target_yaw;
  double current_x;
  double current_y;
  double current_z;
  double current_yaw;
  double qx, qy, qz, qw;

  double pthrust, ithrust, dthrust;
  double phorizontal, ihorizontal, dhorizontal;
  double pyaw, iyaw, dyaw;
  bool ctrl_yaw;
  double base_thrust;

  void control_thrust(double period);
  void control_yaw(double period);
  void control_horizontal(double period);
  
public:

  PidController();
  virtual ~PidController() {

  }

  void notify_angles(double roll, double pitch, double yaw);
  void notify_angle_rates(double roll_rate, double pitch_rate, double yaw_rate);
  void notify_position(double x, double y, double z);
  void notify_velocity(double vx, double vy, double vz);
  void notify_attitude(double x, double y, double z, double w);
  void control(double time_seconds);
  
  void set_target_position(double x, double y, double z);
  void set_target_velocity(double vx, double vy, double vz);
  void set_target_yaw(double yaw);
  void set_time_to_target(double time_to_target); // time to target in seconds
  
  double get_err_pitch();
  double get_err_roll();
  double get_err_z();
  bool initialize();
  std::vector<double> get_positions();
  std::vector<double> get_state();
  std::vector<double> get_ref_state();

  void set_thrust_pid(double p, double i, double d);
  void set_yaw_pid(double p, double i, double d);
  void set_pitch_roll_pid(double p, double i, double d);
  void set_base_thrust(double val);

};

#endif
