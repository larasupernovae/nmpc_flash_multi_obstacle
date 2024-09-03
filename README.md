# NMPC_Flash_Multi_Obstacle

Welcome to **NMPC_Flash_Multi_Obstacle**: a robust framework for Nonlinear Model Predictive Control (NMPC) tailored for autonomous navigation of DJI Matrice 100 and other UAVs equipped with similar black-box systems. Central to this framework is its real-time optimization capability, enabled by the CasADi library and ROS topics and nodes, which efficiently solves the NMPC problem under tight computational constraints. This allows for adaptive and agile navigation even in dynamic environments. Our approach supports a wide range of trajectory types and obstacle configurations, ensuring flexibility and robustness in various testing scenarios.

Constact here if you have any questions [email Lara](mailto:lara.laban@control.lth.se) or write to her on [LinkedIn](https://www.linkedin.com/in/lara-laban-571804212/).

<p align="center">
  <img src="https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/granso_first.gif" alt="First GIF" width="400" style="margin-right: 10px;" />
  <img src="https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/granso_second.gif" alt="Second GIF" width="400" style="margin-right: 10px;" />
</p>
<p align="center">
  <img src="https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/third.gif" alt="Third GIF" width="400" style="margin-right: 10px;" />
  <img src="https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/fourth.gif" alt="Fourth GIF" width="400" />
</p>

**Videos** Explore our demo showcasing the NMPC_Flash_Multi_Obstacle framework in action at Gränsö Slott, Sweden. [![Watch Playlist](https://img.youtube.com/vi/PLumVztiVR9hmHMMD6MF9ELFhX6l3DU6vO/0.jpg)](https://www.youtube.com/playlist?list=PLumVztiVR9hmHMMD6MF9ELFhX6l3DU6vO)

**Authors**: Tommy Persson and Lara Laban, from [LiU](https://liu.se/en/), [WARA-PS](https://portal.waraps.org/page/home/) and [LU](https://www.lunduniversity.lu.se/).

If you use NMPC_Flash_Multi_Obstacle in your research, please cite our paper as follows:

 - [Enhanced Autonomous UAV: Custom Non-linear Model Predictive Control for Robust Obstacle Avoidance in Diverse Environments], Lara Laban, Mariusz Wzorek, Piotr Rudol, Tommy Persson, Björn Olofsson, Yiannis Karayiannidis, Rolf Johansson, IEEE International Conference on Robotics and Automation (ICRA) 2025

```bash
@inproceedings{TBA,
  title={Enhanced Autonomous UAV: Custom Non-linear Model Predictive Control for Robust Obstacle Avoidance in Diverse Environments},
  author={Laban, Lara and Wzorek, Mariusz and Rudol, Piotr and Persson, Tommy and Olofsson, Björn and Karayiannidis, Yiannis and Johansson, Rolf},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  pages={TBA},
  publisher={IEEE},
  status={TBA}
}
```

## Table of Contents

- [Required Equipment and Software Setup](#required-equipment-and-software-setup)
  - [Explanation of Computer and Hardware Requirements](#explanation-of-computer-and-hardware-requirements)
  - [Software Installation and Dependencies: CasADi and ROS](#software-installation-and-dependencies-casadi-and-ros)
  - [Integrating ROS with DJI Matrice 100](#integrating-ros-with-dji-matrice-100)
- [Background Explanation of the Code](#background-explanation-of-the-code)
  - [Explanation of Various Files and Their Usage](#explanation-of-various-files-and-their-usage)
  - [How to Tune the Controller?](#how-to-tune-the-controller)
  - [Code Structure Overview](#code-structure-overview)
- [NMPC: Simulation and Simulator Tests](#nmpc-simulation-and-simulator-tests)
  - [Testing out the Go_to_Point Trajectory](#testing-out-the-go_to_point-trajectory)
  - [Testing out the Spline/Circle_xy/Sinusoidal Trajectories](#testing-out-the-splinecircle_xysinusoidal-trajectories)
  - [Testing out the Rectangle/Hexagon Trajectories](#testing-out-the-rectanglehexagon-trajectories)
  - [Testing out the Step_xyz/Line_xy/Step_z Trajectories](#testing-out-the-step_xyzline_xystep_z-trajectories)
- [NMPC: Simulation and Simulator Tests with Static Obstacles](#nmpc-simulation-and-simulator-tests-with-static-obstacles)
  - [One Obstacle](#one-obstacle)
  - [Multiple Obstacles](#multiple-obstacles)
- [NMPC: Simulation and Simulator Tests with Dynamic Obstacles](#nmpc-simulation-and-simulator-tests-with-dynamic-obstacles-working-at-the-moment-...)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Required Equipment and Software Setup

### Explanation of Computer and Hardware Requirements

For the successful implementation of the NMPC framework described in the paper, the following hardware is required:

- **Computer:** The NMPC algorithm was tested on a system equipped with an Intel® Core™ i7-7567U CPU (aka the NUC) at 3.50GHz. A similar or more powerful CPU is recommended to handle the computational load. This was placed on the DJI Matrice 100. (with enough imagination and money, you can acquire a better CPU and maybe a newer DJI and do a similar take over).

- **Memory:** At least 16GB of RAM is necessary to ensure smooth operation, especially during real-time control tasks.

- **Storage:** A 500GB SSD is recommended for storing the ROS environment, necessary packages, and logged data.

- **UAV:** The framework is specifically tailored for the DJI Matrice 100 UAV, which includes the necessary flight controller and hardware interfaces.

- **Other Equipment:** A Vicon Motion Capture System (for indoor experiments) and GPS modules (for outdoor experiments) are essential for accurate positioning and tracking (the setup scheme is provided in the paper). For testings in the simulator before trying it out in the Real World you will need to connect the DJI UAV to a windows machine (with a DJI simulator), and ssh with your own powerful ThinkPad (or any other good machine that supports the Robot Operating System (ROS)) offload computations (for visualisation) from your NUC - this way you test your flight indoors (do not forget to remove propllers). 

### Software Installation and Dependencies: CasADi and ROS
The NMPC framework relies on several software components:

- **Operating System:** Ubuntu 20.04.6 LTS is the OS for setting up the development environment (you kind of have no choice here since its ROS).

- **ROS (Robot Operating System):** ROS Noetic is used for integrating the UAV with the NMPC framework. Install ROS Noetic by following the official installation guide.

(For a future project: by the time you are reading this I guess ROS might be past tense, so of course take Ubuntu 22.04 LTS version and use ROS2 - for a new project and rewrite my code)

- **CasADi:** This Python library is used for numerical optimization and solving nonlinear programming problems. Install it using:

```bash
pip install casadi
sudo apt-get install coinor-libipopt-dev
sudo apt-get install ros-noetic-realsense2-camera
```

Compile Casadi:
```bash
git clone https://github.com/casadi/casadi.git casadi
cd casadi
mkdir build
cd build
cmake ..
make
sudo make install
sudo pip3 install casadi
```

(I am going to level with you here the installation is a bit more sinister then this, I only remember that if you want to use ma27 or ma97 (instead of mumps) you need to visit [HSL](https://licences.stfc.ac.uk/product/coin-hs) --- this is my reccomendation for hard problems --- dont get discouraged if its a hard to link
I think its something about a library being misspelled (libcoinhsl.so) and not changed in the documentation. - best of luck)

```bash
sudo pip3 install meson
sudo apt install ninja-build libmetis-dev
meson setup builddir --buildtype=release --prefix=/opt/coinhsl
meson compile -C builddir
sudo meson install -C builddir
cd /opt/coinhsl/lib/x86_64-linux-gnu/
sudo ln -s libcoinhsl.so libhsl.so
```

### Integrating ROS with DJI Matrice 100

Install DJI SDK: Download and install the DJI Onboard SDK, which provides the necessary libraries and tools to interface with the Matrice 100. Follow the [DJI SDK documentation](https://developer.dji.com/mobile-sdk/documentation/introduction/index.html).

Set Up ROS Nodes: Implement ROS nodes to handle communication between the NMPC controller and the UAV. This involves publishing control commands and subscribing to telemetry data. [ROS Wikia](http://wiki.ros.org/dji_sdk)  

Configuration: Configure the launch files to ensure the correct initialization of parameters for the Matrice 100. This includes setting up topics for receiving GPS data, IMU readings, and other necessary inputs for the NMPC algorithm.

## Background Explanation of the Code

### Explanation of Various Files and Their Usage

This will be clear as you try the envorment, here are some additional files I provide for future usage.

   - `MPC_lidar_control_WALL_cvxgen_granso.cpp` : Implements the core NMPC algorithm, integrating LIDAR data for obstacle detection and avoidance. Its my old code using a linearized MPC with cvxgen to keep a distance to the wall this paper will never be published so I was like add it here why not? 
   
   - `lidar_python_code_skelet.py`: A Python script skeleton that processes LIDAR data to identify obstacles and calculate safe paths. This is for whoever want to continue my mission of dynamical obstacles, helper code.
   
   - `py_nmpc_dji100_DYNAMIC_OBSTACLE.py`: A Python script for dynamic obstacle avoidance, integrating the NMPC framework with real-time data from the DJI Matrice 100, still not finished also who ever wants to finish good skelet of what you need?

### How to Tune the Controller?

NOTE: Don't forget to adjust ----> self.k = 0.69  # Lift constant  # The more it goes up the lower the thrust <------ as it influences the thrust
and makes a difference how the drone controls the altitude, by adjusting it we have the blue trajectory almost perfectly following the green one


### Code Structure Overview

#### Algorithm 1: NMPC Implementation for UAV

1. **Initialize**:
    - `x = MX.sym('x', 10)`
    - `u = MX.sym('u', 4)`
2. Define system dynamics `f(x, u)`
3. Define objective function `L(x, u, x_ref)`
4. **Set up optimization problem**:
    - Initialize variables: `w`, `w_0`, `lbw`, `ubw`, `g`, `lbg`, `ubg`
    - Set initial state: `X_k = MX.sym('X0', 10)`
    - `w = [X_k]`, `lbw = [x_0]`, `ubw = [x_0]`, `w_0 = [x_0]`
5. **For** `k = 0` to `N-1` **do**:
    1. Define control: `U_k = MX.sym(f'Uk_{k}', 4)`
    2. `w = w ∪ [U_k]`, `lbw = lbw ∪ [u_min]`, `ubw = ubw ∪ [u_max]`
    3. `w_0 = w_0 ∪ [u_0]`
    4. Integrate dynamics: `X_{k+1} = f(X_k, U_k)`
    5. `w = w ∪ [X_{k+1}]`, `lbw = lbw ∪ [-∞]`, `ubw = ubw ∪ [∞]`
    6. `w_0 = w_0 ∪ [0]`
    7. Add equality constraint: `g = g ∪ [X_{k+1} - X_k]`
    8. `lbg = lbg ∪ [0]`, `ubg = ubg ∪ [0]`
6. Define NLP solver with IPOPT
7. **Solve NMPC**:
    - Update `w_0` with `x_k, u_{k-1}`
    - `sol = solver(w_0, lbw, ubw, lbg, ubg)`
    - Extract optimal control `u_opt`
    - Update initial guesses for next iteration

#### Algorithm 2: NMPC Controller Initialization and Operation

1. **Import Libraries**: numpy, pandas, rospy, math, sys
2. **Load Battery Data**:
    - Define the path to the battery data file
    - Read the battery data file into a pandas dataframe
3. **Set Up Command-Line Arguments**:
    - Create a parser for command-line options
    - Add an option for selecting the controller type, default to "FLASH"
    - Add an option for setting the drone's speed, default to 0.35
    - Parse the command-line arguments
4. **ROS Node Initialization**:
    - `rospy.init_node("controller")`
    - `rospy.subscriber("/dji_sdk/battery_state")`
    - `ctrl_pub = rospy.Publisher("dji_sdk/flight_control_setpoint_generic", Joy)`
5. **Callback Functions**:
    - `def battery_callback(data):`
        - Initialize `thrust_adjustment = get_thrust_based_on_battery(data.percentage)`
        - `controller.update_thrust(thrust_adjustment)`
    - `def timer_callback(event):`
        - Enabling visualization in RViz
        - `u_opt = controller.tick()`
        - If `u_opt[0]`:
            - `msg = Joy()`
            - `msg.axes = [u_opt[1], u_opt[2], u_opt[0], u_opt[3]]`
            - `ctrl_pub.publish(msg)`
6. **Start ROS Spin**:
    - `rospy.Timer(rospy.Duration(0.1), timer_callback)`
    - `rospy.spin()`

This a life tip you can use as you wish, warmest reccomendation to use MX instead of SX in problems complex as this, since otherwise you will have issues to use jacobian and similar stuff with matrices (also look at the documentation of CasADi, not on the webpage but inside the code - this might be a game changer at some point).

## NMPC: Simulation and Simulator Tests

30-90Hz by using Explicit Euler (it is set to this inside the NMPC for all at the moment). However, you also have options to use the Runge-Kutta 4 
or maybe the CasADi integrator however even tho this will work in the simulation, in the simulator this will affect sampling and lower your Hz, which will in turn make it impossible to do a real flight. Using Explicit Euler means you are losing accuracy in the trajectory tracking but gaining Hz. 

note to self: 'ma97' HSL solver doesn't seam to speed up stuff even with the parallel computation CPU, the problem is to hard? I guess yes, doesn't matter if the solver is better, if your problem is hard on its own. 

Adding --use_nmpc flag gives you the simulation (`python3 py_nmpc_dji100_OBSTACLE.py --trajectory_type spline --use_nmpc`) without the flag, this same code works for the DJI simulator.

Default settings if something goes wrong, otherwise we are calculating the lenghts of the spline, and adjusting the weights inside the code:

```bash
self.k = 0.76  # Lift constant  # The more it goes up the lower the thrust

self.speed = 0.35 #0.2  # 0.1   # (1 m/s = 3.6 km/h)
self.speed_rviz = 0.35 #0.2  #0.75
self.Duration = 15/self.speed # 75 
self.Duration_rviz = 15/self.speed_rviz # for hexagon # 75 #20   15/0.2 = 75

q_x, q_y, q_z = 1.0, 1.0, 1.0 # 1.0, 1.0, 4.0 # Increased positional weights  
q_vx, q_vy, q_vz = 0.0, 0.0, 0.0  # Kept the same for velocity
q_phi, q_theta, q_psi = 0.8, 0.8, 0.8   # Orientation may not be as critical
q_r = 0.0   # Angular rates are kept the same
```

In cased of crashes in the simulator due to the formulation and complexity of the trajectory, please make sure you adjust the iteration number, the sensitivity and etc. 

Please make sure to change spline to circle_xy/sinusoidal/long_spline/... if you want that trajectory, and also increase the iteratrions in the following 
line of the NMPC if something is crashing (keep in mind what works in the simulation might crash in the DJI simulator testing), for example if 90 iteration seamed to always be enough in the simulation, then when you use the DJI Simulator and simulate GPS (closer to Real World Scenarion) you will experience crahses in that case you go from 90 to 200 iteration in the section below - same goes the other way around, if you want it to run faster make the iterations lower if you need more sampling (and lowering iterations does not result in a crash).

```bash
# Define solver options
opts = {
        'ipopt.print_level': 5, 'print_time': 1, #3 1
        'ipopt.warm_start_init_point': 'yes',  # Tell IPOPT to use warm start
        'ipopt.warm_start_bound_push': 1e-3,
        'ipopt.warm_start_mult_bound_push': 1e-3,
        'ipopt.mu_init': 1e-3,  # Initial value for the barrier parameter
        'ipopt.max_iter': 200,  #90  # Maximum number of iterations
        'ipopt.tol': 1e-2, #2,       # Tolerance for convergence the bigger the number the tighter the tolerance
        'ipopt.acceptable_tol': 1e-3, #3, # Acceptable tolerance for convergence
        'ipopt.linear_solver': 'mumps', # 'mumps' #'ma97' # Linear solver to be used}  # Increased print level
    }
```

### Testing out the go_to_point trajectory - GO TO A SPECIFIC POINT

Based in the trajectory you are using make sure you tune the following:

```bash
if trajectory_type == "go_to_point":
    self.duration_const = 1.0    
    q_x, q_y, q_z = 1.0, 1.0, 10.0 
    #q_phi, q_theta, q_psi = 1.4, 1.4, 0.8   # mozda a mozda i ne
    if use_nmpc==True:
        q_x, q_y, q_z = 1.0, 1.0, 4.0 
```           

#### DJI Simulator testings

**THIS IS VALID FOR ALL FOLLOWING RVIZ VISUALIZATION**:
The yellow sphere represents the obstacles, the green line is the desired trajectory, the blue line is the actual path taken by the UAV, the red line shows the prediction horizon, and the yellow line indicates the path being sent to the controller at time k. Time stemps where used to notice the changes when using the position controller (perfect match with constant speed) and notice the change when using the velocity controller.

If you wish to go to a specific point, the default on is [0, 0, 1], with the following command:

```bash
python3 pycontroller.py --trajectory_type go_to_point --controller FLASH __ns:=/dji1
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/tacka_1.png)

Example of setting the speed and fly to point: (note to self: B-splines solve the issue of having only two points adjust your k for smoothnes
based on how many you have):

Example of going to the following point -x 1.5 -y 1.6 -z 3.6 and the speed to 0.85:

```bash
python3 pycontroller.py --trajectory_type go_to_point --controller FLASH __ns:=/dji1 -x 1.5 -y 1.6 -z 3.6 --speed 0.85
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/tacka_2.png)

(Here we can note that we had an overshoot and the speed was to much for such the UAV, if we use the same command as previously to return:)

```bash
python3 pycontroller.py --trajectory_type go_to_point --controller FLASH __ns:=/dji1
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/tacka_3.png)

(we can note that the overshoot is much smaller, given this we can conclude that 0.35 is the optimal constant speed we need. Remember I am still demonstrating the position controller here only, in the obstacle section we will play around with adjusting the speed)

Visualize in Rviz:

```bash
rosrun rviz rviz -d ./granso.rviz
```

Also if your command line is something like this you are on the right track:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/tacka_4.png)

### Testing out the spline/circle_xy/sinusoidal trajectories

Based in the trajectory you are using make sure you tune the following:
p.s. Also here inside the code you can place your dsired spline and adjust how many points you want your spline to have

```bash
# Define trajectory based on type
if trajectory_type == "spline":
    # Define the 5 points in XYZ space for the spline
    self.num_points=5
    self.k = 0.75  # Lift constant  # The more it goes up the lower the thrust
    #q_x, q_y, q_z = 0.5, 0.9, 0.9 # 1.0, 1.0, 4.0 # Increased positional weights
    q_x, q_y, q_z = 1.0, 1.0, 10.0
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
```

#### Simulation testings

```bash
python3 py_nmpc_dji100_FLASH.py --trajectory_type spline --use_nmpc 
```

3D visualization of the actual vs desired trajectory:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/spline_1.png)

Control Inputs: Thrust [%], Roll [rad], Pitch [rad], Yaw Rate [rad/s]:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/spline_2.png)

States: x, y, z, phi, theta, psi, xdot, ydot, zdot, psidot:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/spline_3.png)

All simulations will present the same 3 Figures, 3D path, control inputs and states, we have ommited further Images due to clarity.

```bash
python3 py_nmpc_dji100_FLASH.py --trajectory_type sinusoidal --use_nmpc 
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/sinusoidal_simulation_picture.png)

```bash
python3 py_nmpc_dji100_FLASH.py --trajectory_type circle_xy --use_nmpc 
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/circle_simulation_picture.png)

#### DJI Simulator testings

```bash
rostopic echo /dji1/dji_sdk/local_position
```

```bash
rosservice call /dji1/dji_sdk/drone_task_control 4
```

```bash
rostopic echo /dji1/dji_sdk/flight_control_setpoint_generic
```

First fly to the point in order to better transtion into the spline. 

```bash
python3 pycontroller.py --trajectory_type go_to_point __ns:=/dji1
```

Then use the spline command (this concept will further be used in the remaining description)

```bash
python3 pycontroller.py --trajectory_type spline --controller FLASH __ns:=/dji1 --speed 0.85
```

Visualize in Rviz:

```bash
rosrun rviz rviz -d ./granso.rviz
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/rviz_spline.png)

```bash
python3 pycontroller.py --trajectory_type go_to_point --controller FLASH __ns:=/dji1 -x 0.0 -y 0.0 -z 3.0
```

```bash
python3 pycontroller.py --trajectory_type sinusoidal --controller FLASH __ns:=/dji1 --speed 0.65
```

Visualize in Rviz:

```bash
rosrun rviz rviz -d ./granso.rviz
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/rviz_sinusoida.png)

```bash
python3 pycontroller.py --trajectory_type go_to_point __ns:=/dji1
```

```bash
python3 pycontroller.py --trajectory_type circle_xy --controller FLASH __ns:=/dji1 --speed 0.65
```

Visualize in Rviz:

```bash
rosrun rviz rviz -d ./granso.rviz
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/circle_dji_picture.png)

### Testing out the rectangle/hexagon trajectories

Based in the trajectory you are using make sure you tune the following:

```bash
if trajectory_type == "rectangle":
    self.length_x = 2.5
    self.length_y = 2.5
    self.k = 0.76  # Lift constant  # The more it goes up the lower the thrust
    q_x, q_y, q_z = 1.0, 1.0, 10.0 # 1.0, 1.0, 4.0 # Increased positional weights  
    if use_nmpc==True:
        q_x, q_y, q_z = 1.0, 1.0, 1.0 

if trajectory_type == "hexagon":
    self.side_length = 6.2 #1.2
    self.k = 0.76  # Lift constant  # The more it goes up the lower the thrust
    q_x, q_y, q_z = 1.0, 1.0, 10.0 # 1.0, 1.0, 4.0 # Increased positional weights    
    if use_nmpc==True:
        q_x, q_y, q_z = 1.0, 1.0, 1.0 
```

#### Simulation testings

```bash
python3 py_nmpc_dji100_FLASH.py --trajectory_type hexagon --use_nmpc 
```
3D visualization of the actual vs desired trajectory:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/hexagon_1.png)

Control Inputs: Thrust [%], Roll [rad], Pitch [rad], Yaw Rate [rad/s]:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/hexagon_2.png)

States: x, y, z, phi, theta, psi, xdot, ydot, zdot, psidot:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/hexagon_3.png)

You will notice we used offsets in the code this concept was utilized to move the hexagon and other shapes and adjustem in our limited Vicon space.

```bash
python3 py_nmpc_dji100_FLASH.py --trajectory_type rectangle --use_nmpc 
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/recatngle_picture.png)


#### DJI Simulator testings

```bash
python3 pycontroller.py --trajectory_type go_to_point --controller FLASH __ns:=/dji1 -x 0.0 -y 0.0 -z 2.5
```

```bash
python3 pycontroller.py --trajectory_type hexagon --controller FLASH __ns:=/dji1
```

Visualize in Rviz:

```bash
rosrun rviz rviz -d ./granso.rviz
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/dji_hexagon_picture.png)

```bash
python3 pycontroller.py --trajectory_type go_to_point --controller FLASH __ns:=/dji1 -x 0.0 -y 0.0 -z 3.0
```

```bash
python3 pycontroller.py --trajectory_type rectangle --controller FLASH __ns:=/dji1
```

Visualize in Rviz:

```bash
rosrun rviz rviz -d ./granso.rviz
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/dji_recatngle_picture.png)

### Testing out the step_xyz/line_xy/step_z trajectories

Based in the trajectory you are using make sure you tune the following:

```bash
if trajectory_type == "step_xyz":
    self.step_height_x = 1.5
    self.step_height_y = 1.5
    self.step_height_z = 1.5
    self.Duration = (abs(self.step_height_x) + abs(self.step_height_y) + abs(self.step_height_z))/self.speed  
    self.Duration_rviz = self.Duration
    self.speed_rviz = self.speed
    if use_nmpc==True:
        q_x, q_y, q_z = 1.0, 1.0, 4.0 

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
```

The images are omitted however the concept is fairly similar to the previous ones. 

## NMPC: Simulation and Simulator Tests with STATIC OBSTACLES

### One Obstacle

In the following section we have the position and velocity controller and the different influences, the velocity controller allows for changing speed and has a smoother following of the B-spline.

#### Simulation testings

```bash
python3 py_nmpc_dji100_OBSTACLE.py --trajectory_type spline --use_nmpc
```
![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/one_obstacle.png)

#### DJI Simulator testings

Depending on what we want the controller to focus on position or velocity we need to uncomment the following: 
```bash
q_x, q_y, q_z = 1.0, 1.0, 10.0
#q_vx, q_vy, q_vz = 2.5, 2.5, 2.5  # FOCUS ON VELOCITY
q_vx, q_vy, q_vz = 0.5, 0.5, 8.5  # FOCUS ON POSITION
```
```bash
python3 pycontroller.py --trajectory_type go_to_point --controller FLASH __ns:=/dji1 -x 18.0 -y 5.0 -z 8.0 --speed 0.5
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/rviz_one_prep.png)

#### Focus on Position Controller / Constant Velocity

```bash
python3 pycontroller.py --trajectory_type spline --controller OBSTACLE --obstacle OBSTACLE __ns:=/dji1 --speed 0.75
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/rviz_one_position.png)

#### Focus on Velocity Controller

```bash
python3 pycontroller.py --trajectory_type spline --controller OBSTACLE --obstacle OBSTACLE __ns:=/dji1 --speed 0.75
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/rviz_one_velocity.png)


### Multiple Obstacles

Multiple obstacle path, where the blue line represents the actual flight path, the yellow dashed line indicates the desired trajectory, and the red spheres depict the obstacles, surrounded by a yellow safety distances.

#### Simulation testings

```bash
python3 py_nmpc_dji100_MULTI_OBSTACLES.py --trajectory_type long_spline --use_nmpc
```

3D visualization of the actual vs desired trajectory:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/obstacles_1.png)
![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/obstacles_1a.png)

Control Inputs: Thrust [%], Roll [rad], Pitch [rad], Yaw Rate [rad/s]:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/obstacles_2.png)

States: x, y, z, phi, theta, psi, xdot, ydot, zdot, psidot:

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/obstacles_3.png)


#### DJI Simulator testings

Depending on what we want the controller to focus on position or velocity we need to uncomment the following: 

```bash
q_x, q_y, q_z = 1.0, 1.0, 10.0
#q_vx, q_vy, q_vz = 2.5, 2.5, 2.5  # FOCUS ON VELOCITY
q_vx, q_vy, q_vz = 4.5, 4.5, 8.5 # FOCUS ON VELOCITY
```
```bash
python3 pycontroller.py --trajectory_type go_to_point --controller FLASH __ns:=/dji1 -x 18.0 -y 5.0 -z 8.0 --speed 0.75
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/rviz_multi_prep.png)

#### Focus on Position Controller / Constant Velocity

Here we observe that there is a time limit on the positions, the UAV realises it cannot go around the obstcle so it goes in front of it.

```bash
python3 pycontroller.py --trajectory_type long_spline --controller MULTI_OBSTACLES --obstacle MULTI_OBSTACLES __ns:=/dji1 --speed 1.65
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/rviz_multi_position.png)

#### Focus on Velocity Controller

However, when we realise the velocities the UAV flies around the spheres making it obvious that the obstacle was avoided, while adjusting the speed in real time.

```bash
python3 pycontroller.py --trajectory_type long_spline --controller MULTI_OBSTACLES --obstacle MULTI_OBSTACLES __ns:=/dji1 --speed 2.25
```

![Alt text](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/src/images_for_README/rviz_multi_velocity.png)


## NMPC: Simulation and Simulator Tests with DYNAMIC OBSTACLES - working at the moment...


## License
This project is licensed under the MIT License. You can view the full license [here](https://github.com/larasupernovae/nmpc_flash_multi_obstacle/raw/main/LICENSE.txt).

For more information about the MIT License, you can visit the [Open Source Initiative](https://opensource.org/licenses/MIT) website. 

## Acknowledgements

We gratefully acknowledge the use of [CasADi](https://web.casadi.org/docs/) for solving non-linear optimization problems within our framework. Our integration with [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu/) (Robot Operating System). facilitates seamless communication and control of the UAV system.