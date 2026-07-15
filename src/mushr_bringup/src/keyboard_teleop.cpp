#include <ros/ros.h>
#include <keyboard_teleop.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <algorithm>


namespace KeyboardTeleop{

    keyboard_teleop::keyboard_teleop(ros::NodeHandle nh){
        ackermann_pub=nh.advertise<ackermann_msgs::AckermannDriveStamped>("/ackermann_cmd", 10);
        enableRawMode();
    }


    void keyboard_teleop::enableRawMode(){
        tcgetattr(STDIN_FILENO, &orig_termios); // getting the current terminal state

        struct termios raw = orig_termios; // making a copy to make changes
        raw.c_lflag &= ~(ECHO | ICANON | ISIG); // turning of echoing/pressing enter/system message internception
        raw.c_cc[VMIN] = 0;                             
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    }

    void keyboard_teleop::disableRawMode(){
         tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios); 
    }

    keyboard_teleop::~keyboard_teleop(){
        disableRawMode();
    }





    void keyboard_teleop::run(){
        ros::Rate loop_rate(30);
        bool quit=false;
        int idle_cycles=0; // loop cycles since the last key arrived
        while(ros::ok()){
            char c;
            bool key_seen=false;
            while(read(STDIN_FILENO, &c, 1) > 0){
                switch(c){
                    case 3: // Ctrl+C arrives as a byte since ISIG is off
                    case 'q': speed=0.0;steering=0.0;quit=true;break;
                    case 'w': speed=drive_speed; key_seen=true; break;
                    case 's': speed=-drive_speed; key_seen=true; break;
                    case 'a': steering=turn_angle; key_seen=true; break;
                    case 'd': steering=-turn_angle; key_seen=true; break;
                    case ' ': speed=0.0;steering=0.0; break;

                }
            }

            // videogame-style: keys auto-repeat while held; once they stop
            // arriving for longer than the OS auto-repeat delay, coast to a stop
            if(key_seen) idle_cycles=0;
            else if(++idle_cycles > key_timeout_cycles){
                speed=0.0;
                steering=0.0;
            }

            ackermann_msgs::AckermannDriveStamped msg;
            msg.header.stamp = ros::Time::now();

            speed=std::max(std::min(speed,1.0f),-1.0f);
            steering=std::max(std::min(steering,1.0f),-1.0f);
            msg.drive.speed = speed;
            msg.drive.steering_angle = steering;
            ackermann_pub.publish(msg);

            if(quit){
                ros::Duration(0.1).sleep(); 
                ros::shutdown();
                break;
            }

            ros::spinOnce();
            loop_rate.sleep();
        }


    }

}