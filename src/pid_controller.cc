#include "pid_controller.h"

#include "tf/transform_datatypes.h"

PidController::PidController() : err_z(0.0), old_err_z(0.0), int_err_z(0.0), d_err_z(0.0),
                                 err_yaw(0.0), old_err_yaw(0.0), int_err_yaw(0.0), d_err_yaw(0.0),
                                 pthrust(1.5), ithrust(0.0019), dthrust(6.0),
                                 old_err_pitch(0.0), old_err_roll(0.0),
                                 target_x(0.0), target_y(0.0), target_z(0.0), target_yaw(0.0),
                                 current_x(0.0), current_y(0.0), current_z(0.0), current_yaw(0.0),
                                 qx(0.0), qy(0.0), qz(0.0), qw(1.0),
                                 phorizontal(2.0), ihorizontal(0.0), dhorizontal(4.0),
                                 pyaw(1.0), iyaw(0.0), dyaw(0.5), ctrl_yaw(true),
                                 base_thrust(37.011) {
  
}

bool PidController::initialize() {
  err_z = 0.0;
  old_err_z = 0.0;
  int_err_z = 0.0;
  d_err_z = 0.0;
  err_yaw = 0.0;
  old_err_yaw = 0.0;
  int_err_yaw = 0.0;
  d_err_yaw = 0.0;
  pthrust = 1.5;
  ithrust = 0.0019;
  dthrust = 6.0;
  old_err_pitch = 0.0;
  old_err_roll = 0.0;
  target_x = 0.0;
  target_y = 0.0;
  target_z = 0.0;
  target_yaw = 0.0;
  current_x = 0.0;
  current_y = 0.0;
  current_z = 0.0;
  current_yaw = 0.0;
  qx = 0.0;
  qy = 0.0;
  qz = 0.0;
  qw = 1.0;
  phorizontal = 2.0;
  ihorizontal = 0.0;
  dhorizontal = 4.0;
  pyaw = 1.0;
  iyaw = 0.0;
  dyaw = 0.5;
  ctrl_yaw = true;
  base_thrust = 37.011;  
  return true;
}

void PidController::control_thrust(double period) {
  err_z = target_z - current_z;
  int_err_z += err_z;
  d_err_z = (err_z - old_err_z)/period;
  double delta = pthrust*err_z + ithrust*int_err_z + dthrust*d_err_z;
  double thrust =  base_thrust + delta;
  double limit1 = 80.0;
  double limit2 = 20.0;
  if (thrust > limit1) {
    thrust = limit1;
  }
  if (thrust < limit2) {
    thrust = limit2;
  }
  // double thrust =  36.0 + delta;
  old_err_z = err_z;
  set_thrust(thrust);
}

void PidController::control_yaw(double period) {
  // ROS_INFO("current_yaw -> target_yaw: %f -> %f",   current_yaw*180.0/M_PI, target_yaw*180.0/M_PI);
  double yaw_rate = 0.0;
  err_yaw = target_yaw - current_yaw;
  int_err_yaw += err_yaw;
  double d_err = (err_z - old_err_z)/period;
  if (ctrl_yaw) {
    yaw_rate = pyaw*err_yaw + dyaw*d_err;
  }
  old_err_yaw = err_yaw;
  set_yaw_rate(yaw_rate);
}

void PidController::control_horizontal(double period) {
  double err_x = target_x - current_x;
  double err_y = target_y - current_y;
  err_pitch = cos(-current_yaw)*err_x - sin(-current_yaw)*err_y;
  err_roll = sin(-current_yaw)*err_x + cos(-current_yaw)*err_y;

  double derr_pitch = (err_pitch - old_err_pitch)/period;
  double derr_roll = (err_roll - old_err_roll)/period;

  double pitch = phorizontal*err_pitch + dhorizontal*derr_pitch;
  double roll = -(phorizontal*err_roll + dhorizontal*derr_roll);

  double limit = 10.0;
  if (pitch > limit) pitch = limit;
  if (pitch < -limit) pitch = -limit;
  if (roll > limit) roll = limit;
  if (roll < -limit) roll = -limit;
    
    
  old_err_pitch = err_pitch;
  old_err_roll = err_roll;
  
  set_pitch(pitch*M_PI/180.0);
  set_roll(roll*M_PI/180.0);
}

void PidController::notify_angles(double roll, double pitch, double yaw) {

}


void PidController::notify_angle_rates(double roll_rate, double pitch_rate, double yaw_rate) {

}


void PidController::notify_position(double x, double y, double z) {
  current_x = x;
  current_y = y;
  current_z = z;
}

void PidController::notify_velocity(double vx, double vy, double vz) {
}

void PidController::notify_attitude(double x, double y, double z, double w) {
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
  current_yaw = yaw;
}

double PidController::get_err_pitch() {
  return err_pitch;
}

double PidController::get_err_roll() {
  return err_roll;
}

double PidController::get_err_z() {
  return err_z;
}


void PidController::control(double time_seconds) {
  control_thrust(time_seconds);
  control_yaw(time_seconds);
  control_horizontal(time_seconds);
}

void PidController::set_target_position(double x, double y, double z) {
  target_x = x;
  target_y = y;
  target_z = z;
  ROS_INFO("set_target_position: %f %f %f", x, y, z);
}

void PidController::set_target_velocity(double vx, double vy, double vz) {
  ROS_INFO("set_target_velocity NOOP: %f %f %f", vx, vy, vz);
}

void PidController::set_target_yaw(double yaw) {
  target_yaw = yaw*M_PI/180.0;
}

void PidController::set_time_to_target(double time_to_target) {

}

void PidController::set_thrust_pid(double p, double i, double d) {
  pthrust = p;
  ithrust = i;
  dthrust = d;
  int_err_z = 0.0;
}

void PidController::set_yaw_pid(double p, double i, double d) {
  pyaw = p;
  iyaw = i;
  dyaw = d;
}

void PidController::set_pitch_roll_pid(double p, double i, double d) {
  phorizontal = p;
  ihorizontal = i;
  dhorizontal = d;
}

void PidController::set_base_thrust(double val) {
  base_thrust = val;
}

std::vector<double> PidController::get_positions() {
  std::vector<double> res;
  res.push_back(current_x);
  res.push_back(current_y);
  res.push_back(current_z);
  res.push_back(target_x);
  res.push_back(target_y);
  res.push_back(target_z);
  return res;
}

std::vector<double> PidController::get_state() {
  std::vector<double> res;
  return res;
}

std::vector<double> PidController::get_ref_state() {
  std::vector<double> res;
  return res;
}
