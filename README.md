## Planning Robot for Existential Robotics Lab Undergrad Group

## About the Robot 
We use a variation of the [Mushr Robot](https://mushr.io), which houses a Lidar(Hokuyo UST-10LX), IMU(Phidget Spatial Precision 1044_0), Jetson Tx2, and motor controller(Focbo Enertion). Onboard the Jetson Tx2, we have Ubuntu 18 and ROS melodic. To run the teleoperate program on the robot we use a keyboard(works not as well) or a controller(not playstation/Nintendo due to lack of native support for these companies's protocols). 


## Getting Started

### Clone the repository
```bash
git clone [<repo-url>](https://github.com/tarunja1ks/Lab-Car-2/)
cd Lab-Car-2
```

### Install ROS Melodic

#### Windows
Try to run Ubuntu 18.04 directly if you can. Otherwise, run a VM — preferably with GPU passthrough.

#### macOS
- **Apple Silicon**: Install ROS Melodic via RoboStack conda, then emulate it through Rosetta since it's x86-only.
- **Intel**: Install via RoboStack conda and run it natively.

If unable to install ROS Melodic you can install ROS Noetic as an alternative, which has more online support for installation. However, beware that you may have python conflicts and minor issues when deploying on the robot you may need to debug. 


### Starting the ROS Server(for both the robot and your personal computer)
After changing directories into the repository and into Catkin_WS you must run ``catkin_make`` to compile all of the code. 

To start the ros server type ``roscore`` in a separate terminal window. 

### Running Teleop on the Robot

To begin running teleop on the robot type ``rosrun mushr_bringup lab-car-2.launch`` and you may use the controller to begin driving the robot. If you prefer to drive the robot with a keyboard or don't have a controller, run ``rosrun mushr_bringup gamecontroller_teleop_node`` in a separate terminal window after running the previous command. 

### Launching Gazebo with the Robot

To begin running teleop on the robot type ``rosrun mushr_gazebo gazebo-sim.launch`` and you may use the controller to begin driving the robot. If you prefer to drive the robot with a keyboard or don't have a controller, run ``rosrun mushr_bringup gamecontroller_teleop_node`` in a separate terminal window after running the previous command. 

## Package Descriptions

| Package | Description |
|---|---|
| **mushr_gazebo** | Gazebo simulation. Ackermann drive plugin, world files, laser→pointcloud conversion, octomap mapping node, keyboard/gamepad teleop, and ground-truth tf publisher. |
| **mushr_bringup** | Real-car bringup on the TX2. Launches the VESC driver and ackermann/odom conversion, plus keyboard and gamepad teleop nodes. |
| **mushr_description** | Robot model — `racecar.urdf` with links, wheels, LIDAR, and camera sensor definitions. |
| **ntfields_planner** | NTFields neural motion planner (in progress — training and map-conversion scripts). |
| **mushr_control** | Empty placeholder (no source yet). |
| **mushr_navigation** | Empty placeholder (no source yet). |

Third-party (vendored): `vesc` (motor driver + ackermann/odom conversion), `realsense-ros`, `pedsim_ros_with_gazebo` (pedestrian sim), `gazebo_ros_pkgs`, `ros_control`, `vision_opencv`, `image_common`, `ackermann_msgs`, `octomap_msgs`, `control_toolbox`, `realtime_tools`, `depthimage_to_laserscan`.

## Topic Descriptions

| Topic | Type | Published by | Subscribed by |
|---|---|---|---|
| `/ackermann_cmd` | `AckermannDrive(Stamped)` | teleop nodes (keyboard, gamepad) | ackermann plugin (sim), `ackermann_to_vesc` (car) |
| `/scan` | `sensor_msgs/LaserScan` | Gazebo LIDAR plugin | `lidar_conversion` |
| `/cloud` | `sensor_msgs/PointCloud2` | `lidar_conversion` | `mapping_node` |
| `/octomap_markers` | `visualization_msgs/MarkerArray` | `mapping_node` | rviz |
| `/odom` | `Float64MultiArray` | ackermann plugin (wheel velocities) | `ekf_node` |
| `/steering_angle` | `std_msgs/Float64` | ackermann plugin | `ekf_node` |
| `/ground_truth_pose` | `geometry_msgs/Pose2D` | `ground_truth.py` | — |
| `/tf` | `tf2_msgs/TFMessage` | `ground_truth.py` (`map`→`base_footprint`), `robot_state_publisher` | `mapping_node`, rviz |
| `/ekf_pose` | `geometry_msgs/Pose2D` | `ekf_node` | — |
| `/gazebo/model_states` | `gazebo_msgs/ModelStates` | Gazebo | `ground_truth.py` |



