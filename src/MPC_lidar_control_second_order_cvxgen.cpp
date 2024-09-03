#include "dji_sdk_demo/demo_local_position_control.h"
#include "dji_sdk/dji_sdk.h"
#include <iostream> 
#include <sstream>
#include <math.h>
#include <std_msgs/Float64.h>
//#include "matplotlibcpp.h"
#include <std_msgs/Float32.h>

#include <qpOASES.hpp> // Add the qpOASES library for solving QP problems
#include <Eigen/Dense>
#include <vector>
#include <Eigen/Sparse>
#include <ros/ros.h>

extern "C" {
    #include "solver.h"
}

// Constants for PI controller
real_t desired_horizontal_distance = 0.86; // 2.400; // Desired horizontal distance from the wall   4 metara
real_t max_distance = 6.86;
real_t desired_velocity = 0.0; // You want the drone to stay still.
real_t max_control_signal = 17; // Maximum control signal value
// 22 m/s (ATTI mode, no payload, no wind) 17 m/s (GPS mode, no payload, no wind) - DJI Matrice 100

real_t current_velocity = 0.382134;
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


class MPCController {
public:
    // Constructor
    MPCController() {
        initializeMPC();
        // Initialize cvxgen's solver
        load_default_data();
    }


    double computeMPCControl(double centeraverageRange) {

        // Call cvxgen's solver
        int solver_status = solve();

        // Check if the solver was successful
        if (solver_status == SOLVER_STATUS_OPTIMAL) {
            // Extract the control input from cvxgen's solution
            double u_opt = vars.u[0];  // Assuming u is the name of the control variable in cvxgen
            return u_opt;
        } else {
            // Handle solver errors (e.g., infeasible, unbounded, etc.)
            std::cerr << "Solver error: " << solver_status << std::endl;
            return 0.0;  // Or some other default/fallback value
        }
    }

    // Destructor
    ~MPCController() {
        // Reset matrices and vectors
        // Add any other cleanup code here if needed
    }

private:
    // Data structs used in QP solver
    Vars vars;
    Params params;
    Workspace work;
    Settings settings;
    // All your variable declarations here
    double dt = 0.1; // Time step

private:
    // All your variable declarations here
    real_t dt = 0.1; // Time step
    int N = 10; // Prediction Horizon Length - represents the number of future time steps over which the MPC optimizes the control actions.
            // Initialize matrices and vectors here
    Eigen::MatrixXd A, B, Q, R, H, Q_expanded, R_expanded, A_bar, B_bar, A_eq, g, lbA, ubA, lb, ub;

    // Define system matrices

    void initializeMPC() {
        // All your initialization code here
        A = Eigen::MatrixXd(2, 2);
        B = Eigen::MatrixXd(2, 1);
        Q = Eigen::MatrixXd(2, 2);
        R = Eigen::MatrixXd(1, 1);
        H  =Eigen::MatrixXd(2*N, 2*N); // QP Hessian  
        g = Eigen::VectorXd(2*N);      // QP gradient
        A_eq = Eigen::MatrixXd(2*N, N + 2*N);
        lbA = Eigen::VectorXd(2*N);    // Lower bounds for A_eq
        ubA = Eigen::VectorXd(2*N);    // Upper bounds for A_eq
        lb = Eigen::VectorXd(N);     // Lower bounds for control signal
        ub = Eigen::VectorXd(N);     // Upper bounds for control signal
        //u_previous = Eigen::VectorXd::Zero(N);

        ROS_INFO("Initialized matrices with appropriate sizes.");
        // Esential for MPC implementation
        A << 1, dt,               // System Dynamics  2x2 position, velocity
            0, 1;
        B << 0.5*dt*dt,           // Control Input Dynamics: 2x1 matrix, there's a single control input affecting both states.
            dt;
        Q << 10, 0,               // (State Cost): 2x2 matrix, used to weigh the importance of state errors in the cost function.
            0, 1;
        R << 1;                   // R Matrix (Control Cost): 1x1 matrix, used to weigh the importance of control effort in the cost function.
    
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

        // Q_expanded = Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(N, N), Q);
        Q_expanded = Eigen::MatrixXd(2*N, 2*N);
        Q_expanded.setZero();  // Initialize to zeros
        for (int i = 0; i < N; ++i) {
            Q_expanded.block(2*i, 2*i, 2, 2) = Q;
        }
        
        R_expanded = R(0,0) * Eigen::MatrixXd::Identity(N, N);
       
        ROS_INFO("Q_expanded matrix dimensions: %ld x %ld", Q_expanded.rows(), Q_expanded.cols());
        ROS_INFO("R_expanded matrix dimensions: %ld x %ld", R_expanded.rows(), R_expanded.cols());
       
        // Control constraints
        real_t u_min = -max_control_signal;
        real_t u_max = max_control_signal;

        ROS_INFO("Control constraints: u_min = %f, u_max = %f", u_min, u_max);

        // Setup QP matrices based on system dynamics, cost matrices, and constraints
        A_bar = Eigen::MatrixXd(2*N, 2);  // A_bar Matrix: 20x2 matrix, which represents
        // the stacked system dynamics over the prediction horizon of 10 time steps.

        B_bar = Eigen::MatrixXd(2*N, N);  // B_bar Matrix: 20x10 matrix, which represents 
        // the influence of control inputs over the prediction horizon.
          
        A_bar.setZero();
        B_bar.setZero();

        for (int i = 0; i < N; i++) {
            A_bar.block(2*i, 0, 2, 2) = matrixPower(A, i+1);
            for (int j = 0; j <= i; j++) {
                B_bar.block(2*i, j, 2, 1) = matrixPower(A, i-j) * B;
            }
        }

        ROS_INFO("A_bar matrix dimensions: %ld x %ld", A_bar.rows(), A_bar.cols());
        ROS_INFO("B_bar matrix dimensions: %ld x %ld", B_bar.rows(), B_bar.cols());
       
        //H = B_bar.transpose() * Q_expanded * B_bar + R_expanded;
        H = 2 * B_bar.transpose() * Q_expanded * B_bar + R_expanded;

        // Regularize the Hessian to ensure it's positive definite
        real_t regularization_term = 1e-6;
        H += Eigen::MatrixXd::Identity(H.rows(), H.cols()) * regularization_term;

        // Check if H is positive definite
        Eigen::EigenSolver<Eigen::MatrixXd> solver(H);
        if ((solver.eigenvalues().real().array() > 0).all()) {
            std::cout << "Hessian H is positive definite." << std::endl;
        } else {
            std::cout << "Hessian H is NOT positive definite." << std::endl;
            std::cout << "Eigenvalues of H: " << solver.eigenvalues().transpose() << std::endl;
        }

        // Print out the Hessian matrix for debugging
        std::cout << "Hessian Matrix H: " << std::endl << H << std::endl;

        A_eq.setZero();
 
        // Fill in the block diagonal structure
        for (int i = 0; i < N; ++i) {
            A_eq.block(2 * i, 2 * i, 2, 2) = A;
            A_eq.block(2 * i, 2 * i + 2, 2, 1) = B;
        }

        real_t regularization_value = 1e-4;
        A_eq += Eigen::MatrixXd::Identity(A_eq.rows(), A_eq.cols()) * regularization_value;

         // Print out the A_eq matrix for debugging
        std::cout << "Matrix A_eq: " << std::endl << A_eq << std::endl;

        ROS_INFO("H matrix dimensions: %ld x %ld", H.rows(), H.cols());
        ROS_INFO("A_eq matrix dimensions: %ld x %ld", A_eq.rows(), A_eq.cols());
        
        if (A_eq.fullPivLu().rank() == A_eq.rows()) {
            std::cout << "Matrix A_eq is full rank." << std::endl;
        } else {
            std::cout << "Matrix A_eq is NOT full rank." << std::endl;
        }

       // Set bounds (assuming constraints on states and control inputs)
    
        for (int i = 0; i < 2*N; i += 2) {
            lbA(i) = 0;               // Lower bound for centeraverageRange
            lbA(i + 1) = u_min; // Lower bound for current_velocity

            ubA(i) = max_distance;     // Upper bound for centeraverageRange
            ubA(i + 1) = u_max; // Upper bound for current_velocity
        }

        // Print out the initial bounds
        for (int i = 0; i < lbA.size(); ++i) {
            std::cout << "Initial lbA[" << i << "] = " << lbA(i) << ", ubA[" << i << "] = " << ubA(i) << std::endl;
            if (lbA(i) > ubA(i)) {
                std::cout << "Warning: Constraint conflict at index " << i << ". lbA > ubA." << std::endl;
            }
        }

        lb.setConstant(u_min);
        ub.setConstant(u_max);

        // Print out the constraints for A_eq
        for (int i = 0; i < lb.size(); ++i) {
            std::cout << "lb[" << i << "] = " << lb(i) << ", ub[" << i << "] = " << ub(i) << std::endl;
            if (lb(i) > ub(i)) {
                std::cout << "Warning: Constraint conflict at index " << i << " for A_eq. lb > ub." << std::endl;
            }
        }

        ROS_INFO("lbA vector dimensions: %ld x %ld", lbA.rows(), lbA.cols());
        ROS_INFO("ubA vector dimensions: %ld x %ld", ubA.rows(), ubA.cols());
        ROS_INFO("lb vector dimensions: %ld x %ld", lb.rows(), lb.cols());
        ROS_INFO("ub vector dimensions: %ld x %ld", ub.rows(), ub.cols());

  }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "node_name"); 
    real_t centeraverageRange = 0.826;
    MPCController mpc;
    ROS_INFO("Starting... MPC");
    real_t control_signal = mpc.computeMPCControl(centeraverageRange);  

    ROS_INFO("Control_signal: %f m/s", control_signal); 

    MPCController mpc2;
    ROS_INFO("Starting... MPC again...");
    real_t control_signal2 = mpc2.computeMPCControl(centeraverageRange);  

    ROS_INFO("Control_signal: %f m/s", control_signal2); 
 
    ros::spin();
    return 0;
}


