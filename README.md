## Planning Robot for Existential Robotics Lab Undergrad Group

## About the Robot
We use a variation of the [Mushr Robot](https://mushr.io), which houses a Lidar(Hokuyo UST-10LX), IMU(Phidget Spatial Precision 1044_0), Jetson Tx2, and motor controller(Focbo Enertion). Onboard the Jetson Tx2, we have Ubuntu 18 and ROS melodic. To run the teleoperate program on the robot we use a keyboard(works not as well) or a controller(not playstation/Nintendo due to lack of native support for these companies's protocols).

## Running the Gazebo Simulation (Windows & Mac)

The simulator runs inside Docker, so you don't need to install ROS, Gazebo, or Ubuntu - Docker gives you a real Ubuntu 18.04 + ROS Melodic environment in a container, and its desktop is streamed to a page in your normal web browser. There's no X server (XQuartz/VcXsrv) to install or configure.

### 1. Install Docker Desktop
- **Windows**: [docs.docker.com/desktop/install/windows-install](https://docs.docker.com/desktop/install/windows-install/) - the installer walks you through enabling WSL2, which it needs.
- **Mac**: [docs.docker.com/desktop/install/mac-install](https://docs.docker.com/desktop/install/mac-install/) - works on both Apple Silicon and Intel. Apple Silicon runs the image through Docker's built-in x86 emulation, so it's slower but works.

Open Docker Desktop and make sure it's running (the whale icon shows steady, not animating) before continuing.

### 2. Clone the repository
```bash
git clone https://github.com/dhruvdotc/Lab-Car-2.git
cd Lab-Car-2
```

### 3. Build and start the simulator
```bash
docker compose up --build
```
The first run downloads a ~3 GB base image and compiles about 20 ROS packages, so expect 10-20 minutes depending on your connection and machine. Leave this terminal running - it's streaming the container's log output. Later runs (without code changes) start in a few seconds.

### 4. Open the simulator in your browser
Go to **http://localhost:6080/vnc.html** and click **Connect**. Give it 10-20 seconds after the container starts for Gazebo to finish booting. You should see:
- Gazebo, with the MuSHR car spawned in a hallway and pedestrians walking around
- RViz, showing the car's sensors (lidar scan, point cloud, octomap)

### 5. Drive the car
Keyboard teleop needs its own interactive terminal, so it's run separately with `docker exec` rather than from the browser tab. In a **new** terminal on your host machine:
```bash
docker exec -it lab-car-gazebo bash -lc "source /opt/ros/melodic/setup.bash && source /home/Lab-Car-2/devel/setup.bash && rosrun mushr_gazebo gazebo_keyboard_teleop"
```
Keys: `w`/`s` drive, `a`/`d` steer, `space` stop, `q` quit.

If you have a gamepad controller (not PlayStation/Nintendo - no native protocol support), use `gazebo_gamecontroller_teleop` instead of `gazebo_keyboard_teleop` in the same command.

### Stopping the simulator
In the terminal running `docker compose up`, press `Ctrl+C`, or from another terminal:
```bash
docker compose down
```

### Troubleshooting
| Symptom | Fix |
|---|---|
| Browser tab is blank / connects then goes gray | Gazebo is still booting - wait ~15s and refresh. Check `docker logs lab-car-gazebo` for errors if it persists. |
| `docker compose up --build` fails partway through `catkin_make` | Check `docker logs lab-car-gazebo` / the build output for the missing package name and open an issue - most likely a new source package needs a matching apt dependency added to `Dockerfile`. |
| Very slow / choppy on Mac | Expected on Apple Silicon (the image is x86-only, run through emulation). Increase Docker Desktop's CPU/memory allocation in Settings → Resources. |
| `port 6080 already in use` | Another instance is already running - run `docker compose down` first. |
| `Conflict. The container name "/lab-car-gazebo" is already in use` | A previous container wasn't cleaned up - run `docker rm -f lab-car-gazebo`, then `docker compose up --build` again. |

## Running on the Real Robot (Jetson TX2, native ROS)

This only applies to the physical car, which runs Ubuntu 18.04 + ROS Melodic natively on its onboard Jetson TX2 - the Docker setup above is for simulation only.

```bash
git clone https://github.com/dhruvdotc/Lab-Car-2.git
cd Lab-Car-2
catkin_make
roscore   # in its own terminal
```

To drive the real car: `roslaunch mushr_bringup lab-car-2.launch`, then in a separate terminal `rosrun mushr_bringup gamecontroller_teleop_node` (or the keyboard equivalent) if you don't have a controller.

## Package Descriptions

| Package | Description |
|---|---|
| **mushr_gazebo** | Gazebo simulation. Ackermann drive plugin, world files, laser→pointcloud conversion, octomap mapping node, ESDF node, keyboard/gamepad teleop, and ground-truth tf publisher. |
| **mushr_bringup** | Real-car bringup on the TX2. Launches the VESC driver and ackermann/odom conversion, plus keyboard and gamepad teleop nodes. |
| **mushr_description** | Robot model - `racecar.urdf` with links, wheels, LIDAR, and camera sensor definitions. |
| **mushr_control** | Empty placeholder (no source yet). |
| **mushr_navigation** | Empty placeholder (no source yet). |

Third-party (vendored): `vesc` (motor driver + ackermann/odom conversion), `realsense-ros` (real-camera driver, not built for the sim image), `pedsim_ros_with_gazebo` (pedestrian sim), `gazebo_ros_pkgs`, `ros_control`, `vision_opencv`, `image_common`, `ackermann_msgs`, `octomap_msgs`, `control_toolbox`, `realtime_tools`, `depthimage_to_laserscan`.

## Topic Descriptions

| Topic | Type | Published by | Subscribed by |
|---|---|---|---|
| `/ackermann_cmd` | `AckermannDrive(Stamped)` | teleop nodes (keyboard, gamepad) | ackermann plugin (sim), `ackermann_to_vesc` (car) |
| `/scan` | `sensor_msgs/LaserScan` | Gazebo LIDAR plugin | `lidar_conversion` |
| `/cloud` | `sensor_msgs/PointCloud2` | `lidar_conversion` | `mapping_node` |
| `/octomap_markers` | `visualization_msgs/MarkerArray` | `mapping_node` | rviz |
| `/esdf_markers` | `visualization_msgs/MarkerArray` | `esdf_node` | rviz |
| `/odom` | `Float64MultiArray` | ackermann plugin (wheel velocities) | `ekf_node` |
| `/steering_angle` | `std_msgs/Float64` | ackermann plugin | `ekf_node` |
| `/ground_truth_pose` | `geometry_msgs/Pose2D` | `ground_truth.py` | - |
| `/tf` | `tf2_msgs/TFMessage` | `ground_truth.py` (`map`→`base_footprint`), `robot_state_publisher` | `mapping_node`, rviz |
| `/ekf_pose` | `geometry_msgs/Pose2D` | `ekf_node` | - |
| `/gazebo/model_states` | `gazebo_msgs/ModelStates` | Gazebo | `ground_truth.py` |
