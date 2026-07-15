#include <ros/ros.h>
#include <keyboard_teleop.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <algorithm>

#ifdef __APPLE__
// the terminal never reports key releases, so on macOS we ask the OS for the
// physical key state each cycle instead of inferring it from auto-repeat
#include <CoreGraphics/CoreGraphics.h>
static bool key_down(CGKeyCode keycode){
    return CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, keycode);
}
// ANSI-layout virtual keycodes
static const CGKeyCode KEY_W=13, KEY_A=0, KEY_S=1, KEY_D=2;
#endif


namespace KeyboardTeleop{

    keyboard_teleop::keyboard_teleop(ros::NodeHandle nh){
        ackermann_pub=nh.advertise<ackermann_msgs::AckermannDrive>("/ackermann_cmd", 10);
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
        const float dt=1.0f/30.0f;
        bool quit=false;
        int idle_cycles=0; // loop cycles since the last key arrived
        int steer_idle_cycles=0; // loop cycles since the last steering key arrived
        while(ros::ok()){
            char c;
            bool key_seen=false;
            bool steer_seen=false;
            while(read(STDIN_FILENO, &c, 1) > 0){
                switch(c){
                    case 3: // Ctrl+C arrives as a byte since ISIG is off
                    case 'q': speed=0.0;steering=0.0;target_speed=0.0;target_steering=0.0;quit=true;break;
                    case 'w': target_speed=drive_speed; key_seen=true; break;
                    case 's': target_speed=-drive_speed; key_seen=true; break;
                    case 'a': target_steering=turn_angle; key_seen=true; steer_seen=true; break;
                    case 'd': target_steering=-turn_angle; key_seen=true; steer_seen=true; break;
                    case ' ': speed=0.0;steering=0.0;target_speed=0.0;target_steering=0.0; break; // instant stop

                }
            }

#ifdef __APPLE__
            (void)key_seen; (void)steer_seen; (void)idle_cycles; (void)steer_idle_cycles;
            if(!quit){
                bool w=key_down(KEY_W), s=key_down(KEY_S);
                bool a=key_down(KEY_A), d=key_down(KEY_D);
                target_speed    = (w==s) ? 0.0f : (w ?  drive_speed : -drive_speed);
                target_steering = (a==d) ? 0.0f : (a ?  turn_angle  : -turn_angle);
            }
#else
            if(key_seen) idle_cycles=0;
            else if(++idle_cycles > key_timeout_cycles){
                target_speed=0.0;
                target_steering=0.0;
            }

            if(steer_seen) steer_idle_cycles=0;
            else if(++steer_idle_cycles > steer_timeout_cycles) target_steering=0.0;
#endif

            speed    += std::max(std::min(target_speed-speed,       speed_ramp*dt), -speed_ramp*dt);
            steering += std::max(std::min(target_steering-steering, steer_ramp*dt), -steer_ramp*dt);

            ackermann_msgs::AckermannDrive msg;
            msg.speed = speed;               // m/s, converted to wheel rad/s by the plugin
            msg.steering_angle = steering;
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