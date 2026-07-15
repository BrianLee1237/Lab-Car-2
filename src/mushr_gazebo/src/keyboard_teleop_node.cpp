#include <ros/ros.h>
#include <keyboard_teleop.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <algorithm>


int main(int argc, char** argv)
{
  ros::init(argc, argv, "keyboard_teleop_node");
  ros::NodeHandle nh;

  KeyboardTeleop::keyboard_teleop keyboard_based_teleop(nh);
  keyboard_based_teleop.run();

  return 0;
}
