#include <ros/ros.h>
#include <gamecontroller_teleop.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <algorithm>


int main(int argc, char** argv)
{
  ros::init(argc, argv, "gamecontroller_teleop_node");
  ros::NodeHandle nh; 

  GameControllerTeleop::gamecontroller_teleop controller_based_teleop(nh);
  controller_based_teleop.run();

  return 0;
}
