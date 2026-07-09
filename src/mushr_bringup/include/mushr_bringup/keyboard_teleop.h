#ifndef KEYBOARD_TELEOP
#define KEYBOARD_TELEOP

#include <ros/ros.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
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
            float speed=0.0;
            float steering=0.0;
            float drive_speed=0.5; // m/s commanded while w/s is held
            float turn_angle=0.34; // rad commanded while a/d is held
            int key_timeout_cycles=18; // ~0.6s at 30Hz, must outlast the OS key auto-repeat delay
            termios orig_termios;
            // ack_command_publisher = nh.advertise< /*msg_type*/>("ackermann_cmd", 10);
            // ros::ros::Subscriber keyboard_input;
            // /*sub_name*/ = nh.subscribe</*msg_type*/>("/*topic_name*/", 10, /*subscribe_callback_name*/);
            


            
    };
}


#endif