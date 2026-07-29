#include <ros/ros.h>
#include <gamecontroller_teleop.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <SDL2/SDL.h>


namespace GameControllerTeleop{

    gamecontroller_teleop::gamecontroller_teleop(ros::NodeHandle nh){
        SDL_Init(SDL_INIT_JOYSTICK);
        if(SDL_NumJoysticks()>0){
            controller=SDL_JoystickOpen(0);
        }
        ackermann_pub=nh.advertise<ackermann_msgs::AckermannDriveStamped>("/ackermann_cmd", 10);
    }

    void gamecontroller_teleop::run(){
        ros::Rate loop_rate(30);


        const float deadzone=0.1f;
        const int axis_speed=1;
        const int axis_steer=2;

        while(ros::ok()){
            SDL_JoystickUpdate();

            if(!controller || !SDL_JoystickGetAttached(controller)){
                if(controller){ SDL_JoystickClose(controller); controller=nullptr; }
                if(SDL_NumJoysticks()>0) controller=SDL_JoystickOpen(0);
                if(!controller) ROS_WARN_THROTTLE(5.0, "no joystick found");
            }

            int16_t rawSteer   = SDL_JoystickGetAxis(controller, axis_steer);
            int16_t rawSpeed   = SDL_JoystickGetAxis(controller, axis_speed);

            speed=-rawSpeed/32768.0f;
            steering=-rawSteer/32768.0f;

            if(std::abs(speed)<deadzone) speed=0.0f;
            if(std::abs(steering)<deadzone) steering=0.0f;

            ackermann_msgs::AckermannDriveStamped msg;
            msg.header.stamp = ros::Time::now();

            speed=std::max(std::min(speed,1.0f),-1.0f)*2;
            steering=std::max(std::min(steering,1.0f),-1.0f)*0.34f;
            msg.drive.speed = speed;
            msg.drive.steering_angle = steering;
            ackermann_pub.publish(msg);

            ros::spinOnce();
            loop_rate.sleep();
        }


    }

}
