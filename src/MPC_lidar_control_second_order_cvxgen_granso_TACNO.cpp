#include "demo_local_position_control.h"
#include "dji_sdk/dji_sdk.h"
#include <demo_hokuyo_sensor_new.h>
#include "coordtrans.h"
#include <iostream> 
#include <sstream>
#include <math.h>
#include <std_msgs/Float64.h>
//#include "matplotlibcpp.h"
#include <std_msgs/Float32.h>

#include <Eigen/Dense> 
#include <vector>
#include <Eigen/Sparse>

//#include "/home/larablackhole/lrs_ws/src/dji/Onboard-SDK-ROS/dji_sdk_demo/src/cvxgen/solver.h" 
#include "cvxgen/solver.h" 

ros::ServiceClient set_local_pos_reference;
ros::ServiceClient sdk_ctrl_authority_service;
ros::ServiceClient drone_task_service;
ros::ServiceClient query_version_service;

ros::Publisher ctrlPosYawPub;

// Global variables for subscribed topics
uint8_t flight_status = 255;
uint8_t display_mode = 255;
uint8_t current_gps_health = 0;
int num_targets = 0;
geometry_msgs::PointStamped local_position;
sensor_msgs::NavSatFix current_gps_position;
sensor_msgs::LaserScan hokuyo_scan; // New variable for Lidar measurements
bool hasLaserScanData = false;
ros::Publisher average_distance_pub;
ros::Publisher hokuyo_scan_publisher;
ros::Publisher control_signal_pub;
ros::Publisher position_pub;
ros::Publisher error_pub;


// Constants for PI controller
double desired_horizontal_distance = 5.32; //6.42; // 3.50 // Desired horizontal distance from the wall   
double max_distance = 10.86;
double desired_velocity = 0.0; // You want the drone to stay still.
double max_control_signal = 1.29; //1.96; // Maximum control signal value
// 22 m/s (ATTI mode, no payload, no wind) 17 m/s (GPS mode, no payload, no wind) - DJI Matrice 100

int consecutiveSmallControlSignals = 0; // Counter for consecutive small control signals
const int MAX_CONSECUTIVE_SMALL_SIGNALS = 10; // Number of consecutive small signals for stabilization
const double CONTROL_SIGNAL_THRESHOLD = 1.5; // Threshold for control signal to be considered small
double current_velocity = 0.0; // Store the current velocity here

// --------------------------------------------------------------
// -------- MODEL PREDICTIVE CONTROL VARIABLES ------------------
// --------------------------------------------------------------

Eigen::MatrixXd matrixPower(const Eigen::MatrixXd& matrix, int n) {
    // Ensure the matrix is square
    if (matrix.rows() != matrix.cols()) {
        throw std::runtime_error("The matrix is not square.");
    }

    // Base cases
    if (n == 0) {
        return Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
    } else if (n == 1) {
        return matrix;
    }

    Eigen::MatrixXd result = Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
    Eigen::MatrixXd base = matrix;

    while (n > 0) {
        // If n is odd, multiply the result by base
        if (n % 2 == 1) {
            result *= base;
        }

        // Square the base and halve n
        base *= base;
        n /= 2;
    }

    return result;
}

// Function to compute the controllability matrix
Eigen::MatrixXd computeControllabilityMatrix(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    int n = A.rows(); // Number of states
    Eigen::MatrixXd C(n, n*B.cols()); 

    for (int i = 0; i < n; ++i) {
        C.block(0, i*B.cols(), n, B.cols()) = matrixPower(A, i) * B;
    }

    return C;
} 

// Function to check if the system is controllable
bool isSystemControllable(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    Eigen::MatrixXd C = computeControllabilityMatrix(A, B);
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(C);
    return lu_decomp.rank() == A.rows();
}

// Data structs used in QP solver
Vars vars;
Params params;
Workspace work;
Settings settings;

class MPCController {
public:
    // Constructor
    MPCController() {
        initializeMPC();
        // Initialize cvxgen's solver settings
        set_defaults();
        setup_indexing();
        settings.verbose = 0; // Turn off verbosity for the solver
    }


    double computeMPCControl(double centeraverageRange) {

        Eigen::VectorXd x0(2);
        x0 << centeraverageRange, current_velocity; // Assuming initial velocity is 0
    
        // Print the current_velocity value
        std::cout << "Current velocity: " << current_velocity << std::endl;
        // Print the current_velocity value
        std::cout << "centeraverageRange " << centeraverageRange << std::endl;
        // Print the value of x0
        std::cout << "x0: [" << x0(0) << ", " << x0(1) << "]" << std::endl;


        // Convert Eigen matrices to arrays for CVXGEN
        for (int i = 0; i < A.rows(); i++) {
            for (int j = 0; j < A.cols(); j++) {
                params.A[i + j * A.rows()] = A(i, j);
            }
        }

        for (int i = 0; i < B.rows(); i++) {
            params.B[i] = B(i, 0);
        }

        for (int i = 0; i < Q.rows(); i++) {
            for (int j = 0; j < Q.cols(); j++) {
                params.Q[i + j * Q.rows()] = Q(i, j);
            }
        }

        for (int i = 0; i < Q_final.rows(); i++) {
            for (int j = 0; j < Q_final.cols(); j++) {
                params.Q_final[i + j * Q_final.rows()] = Q_final(i, j);
            }
        }

        params.R[0] = R(0, 0);
        params.x[0][0] = x0(0);
        params.x[0][1] = x0(1);
        params.u_max[0] = max_control_signal;

        // Call the CVXGEN solver
        solve();
        
        // Extract the optimal control input from the solver's output
        double u_opt = vars.u[0][0]; // Assuming u is a scalar

        return u_opt;
    }


    // Destructor
    ~MPCController() {
        // Reset matrices and vectors
        // Add any other cleanup code here if needed
    }

private:
    // All your variable declarations here
    double dt = 0.1; // Time step

    Eigen::MatrixXd A, B, Q, R, Q_final;

    // Define system matrices

    void initializeMPC() {
        // All your initialization code here
        A = Eigen::MatrixXd(2, 2);
        B = Eigen::MatrixXd(2, 1);
        Q = Eigen::MatrixXd(2, 2);
        R = Eigen::MatrixXd(1, 1);
        Q_final = Eigen::MatrixXd(2, 2);

        ROS_INFO("Initialized matrices with appropriate sizes.");
        // Esential for MPC implementation
        A << 1, dt,               // System Dynamics  2x2 position, velocity
            0, 1;
        B << 0.5*dt*dt,           // Control Input Dynamics: 2x1 matrix, there's a single control input affecting both states.
            dt;
        Q << 10, 0,               // (State Cost): 2x2 matrix, used to weigh the importance of state errors in the cost function.
            0, 1;
        R << 1;                   // R Matrix (Control Cost): 1x1 matrix, used to weigh the importance of control effort in the cost function.
    
        Q_final << 10, 0,               // (State Cost): 2x2 matrix, used to weigh the importance of state errors in the cost function.
            0, 1;

        ROS_INFO("A matrix dimensions: %ld x %ld", A.rows(), A.cols());
        ROS_INFO("B matrix dimensions: %ld x %ld", B.rows(), B.cols());
        ROS_INFO("Q matrix dimensions: %ld x %ld", Q.rows(), Q.cols());
        ROS_INFO("R matrix dimensions: %ld x %ld", R.rows(), R.cols());

        Eigen::EigenSolver<Eigen::MatrixXd> a_solver(A);
        if ((a_solver.eigenvalues().array().abs() <= 1).all()) {
            std::cout << "System matrix A is stable." << std::endl;
        } else {
            std::cout << "System matrix A is NOT stable." << std::endl;
            std::cout << "Eigenvalues of A: " << a_solver.eigenvalues().transpose() << std::endl;
        }

        Eigen::EigenSolver<Eigen::MatrixXd> q_solver(Q);
        if ((q_solver.eigenvalues().real().array() > 0).all()) {
            std::cout << "Matrix Q is positive definite." << std::endl;
        } else {
            std::cout << "Matrix Q is NOT positive definite." << std::endl;
        }

        Eigen::EigenSolver<Eigen::MatrixXd> r_solver(R);
        if ((r_solver.eigenvalues().real().array() > 0).all()) {
            std::cout << "Matrix R is positive definite." << std::endl;
        } else {
            std::cout << "Matrix R is NOT positive definite." << std::endl;
        }

        if (isSystemControllable(A, B)) {
            std::cout << "The system is controllable." << std::endl;
        } else {
            std::cout << "The system is NOT controllable." << std::endl;
        }

  }
};


// --------------------------------------------------------
// Declare target_index as a global variable
int target_index = 0;

std::vector<geometry_msgs::Point> setpoints;

double centerRange = 0.0;
double centeraverageRange = 0.0;
double xCmd = 0.0, yCmd = 0.0, zCmd = 0.0;

struct Target {
    double latitude;
    double longitude;
    double altitude;
    double yaw;
};

const int max_num_targets = 9;  // Adjust the maximum number of targets accordingly
//double desired_distances[max_num_targets] = {6.42, 6.50, 6.60, 6.70, 6.80, 6.90, 7.0, 7.1, 7.2}; // Example distances for each target
//desired_horizontal_distance = desired_distances[target_index];
Target targets[max_num_targets];

// -----------------------------------------------------------------------------------
// OVAJ DEO JE DODAT ZA KOORDINATE 
// -----------------------------------------------------------------------------------

CoordTrans::CoordTrans(std::string id) {
  //locations["granso"] = new Location("granso", 57.7605573519, 16.6827607783, 29.8, 27.9);
  //locations["lund_ctrldep"] = new Location("lund_ctrldep",  55.708809 ,  13.209846, 35.72, 35.72);
  //locations["golf_lund"] = new Location("golf_lund",  55.7108789, 13.1568357, 26.9, 26.9);
  locations["granso_wall"] = new Location("granso_wall", 57.7605754, 16.6839781, 29.8, 27.9);

  if (locations.find(id) == locations.end()) {
    id = "granso_wall";
  }

  std::cout << "Initializing from position: " << id << std::endl;
  
  lat0 = locations[id]->get_latitude();
  lon0 = locations[id]->get_longitude();
  alt0 = locations[id]->get_altitude();

  int zone = utm_zone(lon0, lat0);

  std::ostringstream os;
  os << "+proj=utm +zone=" << zone << " +ellps=WGS84" << " +preserve_units=False";
  std::cout << "Init string: " << os.str() <<  std::endl;

  P = proj_create(PJ_DEFAULT_CTX, os.str().c_str());
  c_in.lpzt.z = 0.0;
  c_in.lpzt.t = 1000000.0;
  
  c_in.lpzt.lam = lon0*M_PI/180.0;
  c_in.lpzt.phi = lat0*M_PI/180.0;
  // c_out = proj_trans(P, PJ_INV, c);
  std::cout << "INPUT: " << lat0 << " " << lon0 << std::endl;
  c_out = proj_trans(P, PJ_FWD, c_in);

  utm_x = c_out.xy.x;
  utm_y = c_out.xy.y;

  std::cout << utm_x << " " << utm_y << std::endl;

  double lat, lon, alt;
  world_to_wgs84(0.0, 0.0, 0.0, lon, lat, alt);

  std::cout << "(0, 0, 0) = " << lat << " " << lon << " " << alt <<  std::endl;
  
}

int CoordTrans::utm_zone(double lon, double lat) {
  int res = (int)((lon + 180.0) / 6.0) + 1;
  if ((lat >= 56.0) && (lat < 64.0) && (lon >= 3.0) && (lon < 12.0)) {
    res = 32;
  }
  return res;
}

void CoordTrans::world_to_wgs84(double x, double y, double z, double & lon, double & lat, double & alt) {
  c_in.xy.x = x + utm_x;
  c_in.xy.y = y + utm_y;
  c_out = proj_trans(P, PJ_INV, c_in);
  lon = c_out.lpzt.lam*180.0/M_PI;
  lat = c_out.lpzt.phi*180.0/M_PI;
  alt = alt0 + z;
}

void CoordTrans::wgs84_to_world(double lon, double lat, double alt, double & x, double & y, double & z) {
  c_in.lpzt.lam = lon*M_PI/180.0;
  c_in.lpzt.phi = lat*M_PI/180.0;
  //  c_in.lpzt.lam = lon;
  //  c_in.lpzt.phi = lat;
  c_out = proj_trans(P, PJ_FWD, c_in);
  x = c_out.xy.x - utm_x;
  y = c_out.xy.y - utm_y;
  z = alt-alt0;
  //std::cout << "wgs84_to_world: " << lon << " " << lat << " " << alt << " -> " << x << " " << y << " " << z
  //            << std::endl;
}

//CoordTrans coordTrans("lund_ctrldep");
CoordTrans coordTrans("granso_wall");

// -----------------------------------------------------------------------------------
// GLAVNI DEO LETA
// -----------------------------------------------------------------------------------

double processHokuyoSensorData();
void hokuyoScanCallback(const sensor_msgs::LaserScan::ConstPtr &msg);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo_local_position_control_node");
    //ros::init(argc, argv, "velocity_plot_node");
    ros::NodeHandle nh;
    // Subscribe to messages from dji_sdk_node
    ros::Subscriber flightStatusSub = nh.subscribe("dji_sdk/flight_status", 10, &flight_status_callback);
    ros::Subscriber displayModeSub = nh.subscribe("dji_sdk/display_mode", 10, &display_mode_callback);
    ros::Subscriber localPosition = nh.subscribe("dji_sdk/local_position", 10, &local_position_callback);
    ros::Subscriber gpsSub = nh.subscribe("dji_sdk/gps_position", 10, &gps_position_callback);
    ros::Subscriber gpsHealth = nh.subscribe("dji_sdk/gps_health", 10, &gps_health_callback);
    ros::Subscriber hokuyo_scan_subscriber = nh.subscribe<sensor_msgs::LaserScan>("/scan", 10, hokuyoScanCallback);
    
    // Declare the subscriber for velocity data
    ros::Subscriber velocity_sub = nh.subscribe<geometry_msgs::Vector3Stamped>("dji_sdk/velocity", 10, velocity_callback);

    //ros::Subscriber kp_adjustment_subscriber = nh.subscribe<std_msgs::Float64>("adjust_kp", 1, kpAdjustmentCallback);
    //ros::Subscriber ki_adjustment_subscriber = nh.subscribe<std_msgs::Float64>("adjust_ki", 1, kiAdjustmentCallback);

    average_distance_pub = nh.advertise<std_msgs::Float32>("average_distance", 10);
    hokuyo_scan_publisher = nh.advertise<sensor_msgs::LaserScan>("hokuyo_scan", 10);

    // Wait until laser scan data is received
    while (!hasLaserScanData)
    {
        ros::spinOnce();
        ros::Duration(0.1).sleep();
    }

    // Publish the control signal
    ctrlPosYawPub = nh.advertise<sensor_msgs::Joy>("dji_sdk/flight_control_setpoint_ENUposition_yaw", 10);

    //ros::Publisher kp_adjustment_publisher = nh.advertise<std_msgs::Float64>("adjust_kp", 1);
    //ros::Publisher ki_adjustment_publisher = nh.advertise<std_msgs::Float64>("adjust_ki", 1);

    // Initialize the publisher in the main() function

    // OVO JE POTREBNO DA BI MOGLA DA VIZUALIZUJES U RQT_PLOT
    control_signal_pub = nh.advertise<std_msgs::Float64>("control_signal", 10);
    position_pub = nh.advertise<geometry_msgs::Point>("position", 10);
    error_pub = nh.advertise<std_msgs::Float64>("error_signal", 10);
    
    // Basic services
    sdk_ctrl_authority_service = nh.serviceClient<dji_sdk::SDKControlAuthority>("dji_sdk/sdk_control_authority");
    drone_task_service = nh.serviceClient<dji_sdk::DroneTaskControl>("dji_sdk/drone_task_control");
    query_version_service = nh.serviceClient<dji_sdk::QueryDroneVersion>("dji_sdk/query_drone_version");
    set_local_pos_reference = nh.serviceClient<dji_sdk::SetLocalPosRef>("dji_sdk/set_local_pos_ref");

    bool obtain_control_result = obtain_control();
    bool takeoff_result;

    // ovo je ono sto se micalo upitno i plasilo me, ali smo isli blizu zida - podesi bolje tacke
    // targets[0] = {55.7109168, 13.1566563, 4.2, 2}; // Target 1
    // targets[1] = {55.7109085, 13.1565842, 4.2, 2}; // Target 2
    // targets[2] = {55.7109240, 13.1564230, 4.2, 2}; // Target 3
    const double yaw_radians = -145.0 * M_PI / 180.0;

    targets[0] = {57.7605816, 16.6839828, 2.6, yaw_radians}; // Target 1
    targets[1] = {57.7605884, 16.6839811, 2.6, yaw_radians}; // Target 2
    targets[2] = {57.7605959, 16.6839741, 2.6, yaw_radians}; // Target 3
    targets[3] = {57.7606029, 16.6839691, 2.6, yaw_radians}; // Target 4
    targets[4] = {57.7606110, 16.6839644, 2.6, yaw_radians}; // Target 5
    targets[5] = {57.7606145, 16.6839469, 2.6, yaw_radians}; // Target 6
    targets[6] = {57.7606204, 16.6839409, 2.6, yaw_radians}; // Target 7
    targets[7] = {57.7606265, 16.6839328, 2.6, yaw_radians}; // Target 8
    targets[8] = {57.7606378, 16.6839194, 2.6, yaw_radians}; // Target 9

    /*
    if (!set_local_position())
    {
        ROS_ERROR("GPS health insufficient - No local frame reference for height. Exiting.");
        return 1;
    }
    */

    if (is_M100())
    {
        ROS_INFO("M100 taking off!");
        takeoff_result = M100monitoredTakeoff();
    }
    else
    {
        ROS_INFO("A3/N3 taking off!");
        takeoff_result = monitoredTakeoff();
    }

    if (takeoff_result)
    {
        //! Enter total number of Targets
        //num_targets = 3;
        //num_targets = setpoints.size();
        //ROS_INFO("Number of targets: %d", num_targets);
        //std::cout << num_targets << std::endl;
        //! Start Mission by setting Target state to 1
        target_index = 0;

        // Call processHokuyoSensorData() before entering the main spin loop
        //processHokuyoSensorData();    // pozovi ga na drugom mestu - timer function poziva ga non stop scan callback
        //ros::spin();
    }
        
    ros::spin();
    return 0;
}


/*!
 * This function is called when local position data is available.
 * In the example below, we make use of two arbitrary targets as
 * an example for local position control.
 *
 */
void local_position_callback(const geometry_msgs::PointStamped::ConstPtr& msg) {
    if (target_index > max_num_targets) {
    // Landing has already been completed, return from the function
    return;
  }

  static ros::Time start_time = ros::Time::now();
  ros::Duration elapsed_time = ros::Time::now() - start_time;
  local_position = *msg;  
  double xCmd, yCmd, zCmd;
  double x_new_out, y_new_out, z_new_out, yaw_new_out;
  sensor_msgs::Joy controlPosYaw;
  bool land_result;

  // Process Lidar data and adjust control signals
  //processHokuyoSensorData();

  // Down sampled to 50Hz loop
  if (elapsed_time > ros::Duration(0.02)) {
      start_time = ros::Time::now();
      // Here, the code checks the elapsed time to ensure that the loop runs at approximately 50Hz.

      if ((current_gps_health > 1) && (target_index <= max_num_targets)) {
            if (target_index < max_num_targets) {      
              Target currentTarget = targets[target_index];
              setTarget_novo(currentTarget.latitude, currentTarget.longitude, currentTarget.altitude,
                             currentTarget.yaw, x_new_out, y_new_out, z_new_out, yaw_new_out);
              setTarget(x_new_out, y_new_out, z_new_out, yaw_new_out);
              ROS_INFO("Moving towards target %d: x_out = %f, y_out = %f, z_out = %f",
                       target_index + 1, x_new_out, y_new_out, z_new_out);
              local_position_ctrl(xCmd, yCmd, zCmd);
              ROS_INFO("Moving towards target %d: x_new_out = %f, y_new_out = %f, z_new_out = %f",
                       target_index + 1, xCmd, yCmd, zCmd);              
            }
      } else {
          ROS_INFO("Cannot execute Local Position Control");
          ROS_INFO("Not enough GPS Satellites");
          //target_index = 0; // Stop the mission
          // Check if all targets have been completed
      }

    // Check if all targets have been completed
    if (target_index == max_num_targets - 1) {
      // Perform landing after completing all targets
      if (current_gps_health > 1) {
        land_result = M100monitoredLanding();
        if (land_result) {
          ROS_INFO("All targets completed. Landing successful!");
        } else {
          ROS_ERROR("Failed to land after completing all targets.");
        }
      } else {
        ROS_ERROR("Cannot execute landing. Not enough GPS Satellites.");
      }
    }
  }
}

void velocity_callback(const geometry_msgs::Vector3Stamped::ConstPtr& msg) {
    // Handle the velocity data here
    // Store the current velocity
    current_velocity = msg->vector.x;
    // For example, to print the x-component of the velocity:
    //ROS_INFO("Velocity in x: %f", msg->vector.y);
}

// -----------------------------------------------------------------------------------
// POD FUNKCIJA ZA KOORDINATE GDE TREBA DA LETI
// -----------------------------------------------------------------------------------

//CoordTrans coordTrans("lund_ctrldep");

void setTarget_novo(double latitude, double longitude, double altitude, double yaw, double &x_new_out, double &y_new_out, double &z_new_out, double &yaw_new_out)
{  
  // Define the reference point
  double reference_latitude = current_gps_position.latitude;
  double reference_longitude = current_gps_position.longitude;
  double reference_altitude = current_gps_position.altitude;
  
  // Print the reference and target latitude for debugging
  //ROS_INFO("Reference Latitude = %f, Target Latitude = %f", reference_altitude, altitude);
   
  // Convert target latitude and longitude to ENU coordinates
  double deltaX_ref, deltaY_ref, deltaZ_ref;
  coordTrans.wgs84_to_world(reference_longitude, reference_latitude, reference_altitude, deltaX_ref, deltaY_ref, deltaZ_ref);

  // Convert target latitude and longitude to ENU coordinates
  double deltaX, deltaY, deltaZ;
  coordTrans.wgs84_to_world(longitude, latitude, altitude, deltaX, deltaY, deltaZ);

  // Assign the ENU coordinates to the output variables
  x_new_out = deltaX;
  y_new_out = deltaY;
  //ROS_INFO("deltaZ = %f", deltaZ);
  //ROS_INFO("deltaZ_ref = %f", deltaZ_ref);
  z_new_out = altitude;
  yaw_new_out = yaw;

  // Print the calculated ENU offsets for debugging
  //ROS_INFO("x_new_out = %f, y_new_out = %f, z_new_out = %f", x_new_out, y_new_out, z_new_out);

}

void local_position_ctrl(double &xCmd, double &yCmd, double &zCmd)
{
  Target currentTarget = targets[target_index];   // Declare currentTarget here
  //processHokuyoSensorData();
  xCmd = target_offset_x - local_position.point.x;
  yCmd = target_offset_y - local_position.point.y;
  zCmd = target_offset_z;

  sensor_msgs::Joy controlPosYaw;
  controlPosYaw.axes.push_back(xCmd);
  controlPosYaw.axes.push_back(yCmd);
  controlPosYaw.axes.push_back(zCmd);
  controlPosYaw.axes.push_back(target_yaw);
  ctrlPosYawPub.publish(controlPosYaw);

  // 0.1m or 10cms is the minimum error to reach target in x y and z axes.
  // This error threshold will have to change depending on aircraft/payload/wind conditions.
  //if (((std::abs(xCmd)) < 0.1) && ((std::abs(yCmd)) < 0.1) &&
  //    (local_position.point.z > (target_offset_z - 0.1)) && (local_position.point.z < (target_offset_z + 0.1)) &&
  //    (centerRange >= desired_horizontal_distance) && (std::abs(target_yaw) < 0.1)) {
  //  if (target_set_state <= num_targets) {
  //    ROS_INFO("%d of %d target(s) complete", target_set_state, num_targets);
  //    target_set_state++;
  //  }
  //  else {
  //    target_set_state = 0;
  //  }
 // bool asLongasProblem = (std::abs(centeraverageRange - desired_horizontal_distance) > 0.2);
  bool asLongasProblem = centeraverageRange <= desired_horizontal_distance;

  //&& (consecutiveSmallControlSignals >= MAX_CONSECUTIVE_SMALL_SIGNALS); // Adjust the threshold (0.2) as needed

  // Check if the drone is close to the desired horizontal distance
  // bool isCloseToDesiredDistance = std::abs(centeraverageRange - desired_horizontal_distance) < 0.1 && centeraverageRange > desired_horizontal_distance;
  //bool isCloseToDesiredDistance = std::abs(centeraverageRange - desired_horizontal_distance) < 0.1;
  
  // Check if the drone's yaw is almost directly forward
  //bool isYawStable = std::abs(target_yaw) < 0.1;

  // Check if the drone's x and y movements are minimal, indicating it has stabilized
  bool isStablePosition = std::abs(xCmd) < 0.1;
  // Check if there are more targets to process
  bool hasMoreTargets = target_index < max_num_targets;
  
  ROS_INFO("Before the control signal: xCmd = %f, yCmd = %f, zCmd = %f", xCmd, yCmd, zCmd);
  ROS_INFO("Horizontal distance to the wall at the moment: %f meters \n", centeraverageRange);
  //ROS_INFO("isCloseToDesiredDistance aka error: %f meters", (centeraverageRange - desired_horizontal_distance));
  // If the drone is close to the desired distance but not in a stable position, adjust its position
  if (asLongasProblem) {
      // Send control signal to yCmd to adjust its position
      double controlSignal = processHokuyoSensorData();
      xCmd += controlSignal;

      if (std::abs(controlSignal) < CONTROL_SIGNAL_THRESHOLD) {
        consecutiveSmallControlSignals++;
      } else {
        consecutiveSmallControlSignals = 0; // Reset the counter if the control signal is not small 
      }
      ROS_INFO("After the control signal: xCmd = %f, yCmd = %f, zCmd = %f", xCmd, yCmd, zCmd);
      ROS_INFO("Consecutive small signals count: %d", consecutiveSmallControlSignals);
      ROS_INFO("Sending a control signal. Waiting for stabilization...");
       //Introduce a delay to allow the drone to stabilize after adjusting its position
      ros::Duration(5).sleep(); // time you want the drone to wait for stabilization 500ms
  }

  //ROS_INFO("isStablePosition abs(xCmD)  %f", std::abs(xCmd));
  //ROS_INFO("isStablePosition abs(yCmD)  %f", std::abs(yCmd));
  //ROS_INFO("isYawStable %f", std::abs(target_yaw));

  // If the drone is in a stable position and there are more targets to process, move to the next target
  if (!asLongasProblem && isStablePosition && hasMoreTargets) {
      ROS_INFO("Target %d reached. Moving to the next target.", target_index + 1);
      ROS_INFO("Target %d reached. Waiting for stabilization...", target_index + 1);
      ros::Duration(3).sleep(); // Add a delay for stabilization
    
      target_index++;  // Update target_index for the next target
      ROS_INFO("target_index = %d", target_index);
  } else if (target_index == max_num_targets - 1) {
      ROS_INFO("All targets completed.");
      ros::Duration(0.1).sleep();
      return; // Exit the loop for the current target
  }

  // Calculate control_signal, velocity, and position
  geometry_msgs::Point position_msg;
  position_msg.x = local_position.point.x;
  position_msg.y = local_position.point.y;
  position_msg.z = local_position.point.z;
  position_pub.publish(position_msg);
}


 // }
 // else {
 //   // If the current horizontal distance is not yet reached, hold the drone's position.
 //   ROS_INFO("Waiting to reach the desired horizontal distance.");
    // Reset the control commands when the condition is not met
 //   xCmd = 0.0;
 //   yCmd = 0.0;
 //   zCmd = 0.0;
 //   target_yaw = 0.0;
 //   controlPosYaw.axes.push_back(xCmd);
 //   controlPosYaw.axes.push_back(yCmd);
 //   controlPosYaw.axes.push_back(zCmd);
 //   controlPosYaw.axes.push_back(target_yaw);
//    ctrlPosYawPub.publish(controlPosYaw);





// -----------------------------------------------------------------------------------
// GLAVNA FUNKCIJA ZA PI KONTROLER I MERENJA SA LIDAR SENZORA 
// -----------------------------------------------------------------------------------

double processHokuyoSensorData() {

    if (!hasLaserScanData) {
        ROS_WARN("No laser scan data available.");
        return 0.0;
    }

    if (target_set_state > num_targets)
    {
        // Landing has already been completed, return from the function
        return 0.0;
    }

    //while (true) {
        // Calculate the control signal based on the PI controller
    double error = centeraverageRange - desired_horizontal_distance;
        
        // integral_error += error;

    ROS_INFO("Horizontal avg distance to the wall at the moment: %f meters", centeraverageRange);
    ROS_INFO("Greska: %f meters", error);
    
    // Call the MPC controller function to get the control input
    //double control_signal = model_predictive_control(target_offset_x, target_offset_y, target_offset_z, error);
    MPCController mpc;
    ROS_INFO("Starting... MPC");
    double control_signal = mpc.computeMPCControl(centeraverageRange);  

    ROS_INFO("Control_signal: %f m/s", control_signal); 

    // za crtanje plota
    //control_signals.push_back(control_signal);

    // control signal rqt_plot
    std_msgs::Float64 control_signal_msg;
    control_signal_msg.data = control_signal;
    control_signal_pub.publish(control_signal_msg);

    std_msgs::Float64 error_msg;
    error_msg.data = error; 
    error_pub.publish(error_msg);

        // Check if the exit condition is met
        //if (std::abs(error) < 0.1) {
        //    break; // Exit the while loop
        //}
        
        // You might want to introduce a small delay here to control the loop rate
        //ros::Duration(0.1).sleep();
    //}    
    return control_signal;  // Return the calculated control signal
}


// Lidar callback function
void hokuyoScanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{
    hokuyo_scan = *msg;         // ako ga pozivam 10Hz?? proveri koliko daje Hz da znas da li oces update - donesi odluku na osnovu ovoga
    hasLaserScanData = true;
    
    static ros::Time last_callback_time = ros::Time::now();
    ros::Time current_time = ros::Time::now();
    double elapsed_time = (current_time - last_callback_time).toSec();
    last_callback_time = current_time;

    // Find the index of the center beam
    //int centerIndex = hokuyo_scan.ranges.size() / 2;

    // Get the range value of the center beam
    //centerRange = hokuyo_scan.ranges[centerIndex];

    // Check if centerRange is 0.001 and set it to 7.5 meters
    //if (centerRange < 0.05) {
    //    centerRange = 7.5;
    //}

    //static bool printed_max_range = false;
    //if (!printed_max_range) {
    //    ROS_INFO("Hokuyo maximum range: %f meters", hokuyo_scan.range_max);
    //    printed_max_range = true;
    //}

    // Print out all of the ranges the Hokuyo scan gets using ROS_DEBUG
    //for (size_t i = 0; i < hokuyo_scan.ranges.size(); ++i) {
    //    ROS_INFO("Range at index %zu: %f meters", i, hokuyo_scan.ranges[i]);
    //}

    // Find the index of the center beam
    int centerIndex = hokuyo_scan.ranges.size() / 2;

    // Define the angle range for averaging (45 degrees on each side)
    // double angle_range = M_PI / 4.0; // 45 degrees in radians if podeljeno sa 4.0
    double angle_range = 16.0 * (M_PI / 180.0);  //- za 5 stepeni sa svake strane ukupno 10, za 20 menjas sa 20, itd


    // Calculate the number of beams within the angle range
    int num_beams_within_range = std::ceil(angle_range / hokuyo_scan.angle_increment);

    // Calculate the indices of the beams at the edges of the angle range
    int left_index = centerIndex - num_beams_within_range;
    int right_index = centerIndex + num_beams_within_range;

    // Ensure indices are within valid range
    left_index = std::max(left_index, 0);
    right_index = std::min(right_index, static_cast<int>(hokuyo_scan.ranges.size()) - 1);

    // Calculate the sum of distances within the angle range
    double sum_distances = 0.0;
    for (int i = left_index; i <= right_index; ++i) {
        sum_distances += hokuyo_scan.ranges[i];
    }

    // Calculate the average distance within the angle range
    double average_distance = sum_distances / (right_index - left_index + 1);

    // Get the range value of the center beam
    centeraverageRange = average_distance;

    // Check if average_distance is less than 0.05 and set it to 7.5 meters
    if (centeraverageRange < 0.05) {
        centeraverageRange = 6.5;   // 7.5
    }

    // ROS_INFO("Average horizontal distance to the wall: %f meters", average_distance);
    
    std_msgs::Float32 average_distance_msg;
    average_distance_msg.data = average_distance;
    average_distance_pub.publish(average_distance_msg);    

    // Publish the received laser scan data for visualization in RViz
    hokuyo_scan.header.stamp = ros::Time::now();
    hokuyo_scan.header.frame_id = "laser_frame";  // base_laser 

    // Publish the laser scan data on the "/hokuyo_scan" topic
    hokuyo_scan_publisher.publish(hokuyo_scan);
    // Assuming the Hokuyo sensor is mounted horizontally, the centerRange represents
    // the horizontal distance to the wall.
    //ROS_INFO("Horizontal distance to the wall: %f meters", centerRange);

} 

/*
// Lidar callback function
void hokuyoScanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{
    hokuyo_scan = *msg;         // ako ga pozivam 10Hz?? proveri koliko daje Hz da znas da li oces update - donesi odluku na osnovu ovoga
    hasLaserScanData = true;
    
    static ros::Time last_callback_time = ros::Time::now();
    ros::Time current_time = ros::Time::now();
    double elapsed_time = (current_time - last_callback_time).toSec();
    last_callback_time = current_time;

    // Find the index of the center beam
    int centerIndex = hokuyo_scan.ranges.size() / 2;

    // Get the range value of the center beam
    //centerRange = hokuyo_scan.ranges[centerIndex];

    centeraverageRange = hokuyo_scan.ranges[centerIndex];

    // Assuming the Hokuyo sensor is mounted horizontally, the centerRange represents
    // the horizontal distance to the wall.
    //ROS_INFO("Horizontal distance to the wall: %f meters", centerRange);

        // Check if average_distance is less than 0.05 and set it to 7.5 meters
    if (centeraverageRange < 0.05) {
        centeraverageRange = 0.5;   // 7.5
    }

    // ROS_INFO("Average horizontal distance to the wall: %f meters", average_distance);
    
    int average_distance = centerIndex;

    std_msgs::Float32 average_distance_msg;
    average_distance_msg.data = average_distance;
    average_distance_pub.publish(average_distance_msg);    

    // Publish the received laser scan data for visualization in RViz
    hokuyo_scan.header.stamp = ros::Time::now();
    hokuyo_scan.header.frame_id = "laser_frame";  // base_laser 

    // Publish the laser scan data on the "/hokuyo_scan" topic
    hokuyo_scan_publisher.publish(hokuyo_scan);
    // Assuming the Hokuyo sensor is mounted horizontally, the centerRange represents
    // the horizontal distance to the wall.
    //ROS_INFO("Horizontal distance to the wall: %f meters", centerRange);


}
*/










// -------------------------------------------------------------------------------------------
// ODAVDE KRECU SVE POTREBNE ZA AUTORIZACIJU, POLETANJE, SLETANJE...ITD.
// -------------------------------------------------------------------------------------------

bool takeoff_land(int task)
{
  dji_sdk::DroneTaskControl droneTaskControl;

  droneTaskControl.request.task = task;

  drone_task_service.call(droneTaskControl);

  if(!droneTaskControl.response.result)
  {
    ROS_ERROR("takeoff_land fail");
    return false;
  }

  return true;
}

bool obtain_control()
{
  dji_sdk::SDKControlAuthority authority;
  authority.request.control_enable=1;
  sdk_ctrl_authority_service.call(authority);

  if(!authority.response.result)
  {
    ROS_ERROR("obtain control failed!");
    return false;
  }

  return true;
}

bool is_M100()
{
  dji_sdk::QueryDroneVersion query;
  query_version_service.call(query);

  if(query.response.version == DJISDK::DroneFirmwareVersion::M100_31)
  {
    return true;
  }

  return false;
}


void gps_position_callback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
  current_gps_position = *msg;
}

void gps_health_callback(const std_msgs::UInt8::ConstPtr& msg) {
  current_gps_health = msg->data;
}

void flight_status_callback(const std_msgs::UInt8::ConstPtr& msg)
{
  flight_status = msg->data;
}

void display_mode_callback(const std_msgs::UInt8::ConstPtr& msg)
{
  display_mode = msg->data;
}


/*!
 * This function demos how to use the flight_status
 * and the more detailed display_mode (only for A3/N3)
 * to monitor the take off process with some error
 * handling. Note M100 flight status is different
 * from A3/N3 flight status.
 */
bool
monitoredTakeoff()
{
  ros::Time start_time = ros::Time::now();

  if(!takeoff_land(dji_sdk::DroneTaskControl::Request::TASK_TAKEOFF)) {
    return false;
  }

  ros::Duration(0.01).sleep();
  ros::spinOnce();

  // Step 1.1: Spin the motor
  while (flight_status != DJISDK::FlightStatus::STATUS_ON_GROUND &&
         display_mode != DJISDK::DisplayMode::MODE_ENGINE_START &&
         ros::Time::now() - start_time < ros::Duration(5)) {
    ros::Duration(0.01).sleep();
    ros::spinOnce();
  }

  if(ros::Time::now() - start_time > ros::Duration(5)) {
    ROS_ERROR("Takeoff failed. Motors are not spinnning.");
    return false;
  }
  else {
    start_time = ros::Time::now();
    ROS_INFO("Motor Spinning ...");
    ros::spinOnce();
  }


  // Step 1.2: Get in to the air
  while (flight_status != DJISDK::FlightStatus::STATUS_IN_AIR &&
         (display_mode != DJISDK::DisplayMode::MODE_ASSISTED_TAKEOFF || display_mode != DJISDK::DisplayMode::MODE_AUTO_TAKEOFF) &&
         ros::Time::now() - start_time < ros::Duration(20)) {
    ros::Duration(0.01).sleep();
    ros::spinOnce();
  }

  if(ros::Time::now() - start_time > ros::Duration(20)) {
    ROS_ERROR("Takeoff failed. Aircraft is still on the ground, but the motors are spinning.");
    return false;
  }
  else {
    start_time = ros::Time::now();
    ROS_INFO("Ascending...");
    ros::spinOnce();
  }

  // Final check: Finished takeoff
  while ( (display_mode == DJISDK::DisplayMode::MODE_ASSISTED_TAKEOFF || display_mode == DJISDK::DisplayMode::MODE_AUTO_TAKEOFF) &&
          ros::Time::now() - start_time < ros::Duration(20)) {
    ros::Duration(0.01).sleep();
    ros::spinOnce();
  }

  if ( display_mode != DJISDK::DisplayMode::MODE_P_GPS || display_mode != DJISDK::DisplayMode::MODE_ATTITUDE)
  {
    ROS_INFO("Successful takeoff!");
    start_time = ros::Time::now();
  }
  else
  {
    ROS_ERROR("Takeoff finished, but the aircraft is in an unexpected mode. Please connect DJI GO.");
    return false;
  }

  return true;
}


/*!
 * This function demos how to use M100 flight_status
 * to monitor the take off process with some error
 * handling. Note M100 flight status is different
 * from A3/N3 flight status.
 */

bool
M100monitoredTakeoff()
{
  ros::Time start_time = ros::Time::now();

  float home_altitude = current_gps_position.altitude;
  if(!takeoff_land(dji_sdk::DroneTaskControl::Request::TASK_TAKEOFF))
  {
    return false;
  }

  ros::Duration(0.01).sleep();
  ros::spinOnce();

  // Step 1: If M100 is not in the air after 10 seconds, fail.
  while (ros::Time::now() - start_time < ros::Duration(10))
  {
    ros::Duration(0.01).sleep();
    ros::spinOnce();
  }

  if(flight_status != DJISDK::M100FlightStatus::M100_STATUS_IN_AIR ||
     current_gps_position.altitude - home_altitude < 1.0)
  {
    ROS_ERROR("Takeoff failed.");
    return false;
  }
  else
  {
    start_time = ros::Time::now();
    ROS_INFO("Successful takeoff!");
    ros::spinOnce();
  }
  return true;
}

/*

bool M100monitoredTakeoff()
{
    ros::Time start_time = ros::Time::now();

    // Send the takeoff command without checking its success
    takeoff_land(dji_sdk::DroneTaskControl::Request::TASK_TAKEOFF);

    ros::Duration(0.01).sleep();
    ros::spinOnce();

    // Wait for 10 seconds without checking if the drone is in the air
    while (ros::Time::now() - start_time < ros::Duration(10))
    {
        ros::Duration(0.01).sleep();
        ros::spinOnce();
    }

    // Log the takeoff status without checking conditions
    ROS_INFO("Takeoff command sent!");

    return true;
}

*/

/*!
 * This function demos how to use M100 flight_status
 * to monitor the landing process with some error
 * handling. Note M100 flight status is different
 * from A3/N3 flight status.
 */

bool M100monitoredLanding()
{
  ros::Time start_time = ros::Time::now();
  float target_altitude = 0.0; // Set the desired landing altitude

  if (!takeoff_land(dji_sdk::DroneTaskControl::Request::TASK_LAND))
  {
    return false;
  }

  ros::Duration(0.01).sleep();
  ros::spinOnce();

  // Step 1: If M100 is still in the air after 10 seconds, fail.
  while (ros::Time::now() - start_time < ros::Duration(10))
  {
    ros::Duration(0.01).sleep();
    ros::spinOnce();
  }

  if (flight_status != DJISDK::M100FlightStatus::M100_STATUS_ON_GROUND ||
      current_gps_position.altitude > target_altitude)
  {
    ROS_ERROR("Landing failed.");
    return false;
  }
  else
  {
    ROS_INFO("Successful landing!");
    ros::spinOnce();
  }
  
  return true;
}

bool set_local_position()
{
  dji_sdk::SetLocalPosRef localPosReferenceSetter;
  set_local_pos_reference.call(localPosReferenceSetter);

  return (bool)localPosReferenceSetter.response.result;
}
