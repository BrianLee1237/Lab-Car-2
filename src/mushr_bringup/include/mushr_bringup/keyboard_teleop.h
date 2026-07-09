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
            termios orig_termios;
            // ack_command_publisher = nh.advertise< /*msg_type*/>("ackermann_cmd", 10);
            // ros::ros::Subscriber keyboard_input;
            // /*sub_name*/ = nh.subscribe</*msg_type*/>("/*topic_name*/", 10, /*subscribe_callback_name*/);
            


            
    };
}


#endif