#ifndef KEYBOARD_TELEOP
#define KEYBOARD_TELEOP

#include <ros/ros.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

namespace KeyboardTeleop{

    class keyboard_teleop{

        public:
            keyboard_teleop(ros::NodeHandle nh);
            ~keyboard_teleop();
            void run();

        private:
            ros::Publisher ackermann_pub;
            void enableRawMode();
            void disableRawMode();
            float speed=0.0;           // published command, glides toward target
            float steering=0.0;
            float target_speed=0.0;    // set by keys, zeroed by timeouts
            float target_steering=0.0;
            float drive_speed=2.0;     // m/s commanded while w/s is held
            float turn_angle=0.34;     // rad commanded while a/d is held
            float speed_ramp=3.0;      // m/s^2 accel/decel toward target_speed
            float steer_ramp=2.0;      // rad/s toward target_steering
            int key_timeout_cycles=24;   // ~0.8s at 30Hz, outlasts the initial key-repeat delay
            int steer_timeout_cycles=24; // same, but only a/d refresh it so steering re-centers on its own
            termios orig_termios;
            // ack_command_publisher = nh.advertise< /*msg_type*/>("ackermann_cmd", 10);
            // ros::ros::Subscriber keyboard_input;
            // /*sub_name*/ = nh.subscribe</*msg_type*/>("/*topic_name*/", 10, /*subscribe_callback_name*/);
            


            
    };
}


#endif