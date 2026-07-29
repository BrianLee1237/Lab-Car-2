#include <ros/ros.h>
#include <gamecontroller_teleop.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <SDL2/SDL.h>


namespace GameControllerTeleop{

    static volatile sig_atomic_t g_quit=0;
    static void sigint_handler(int){ g_quit=1; }

    gamecontroller_teleop::gamecontroller_teleop(ros::NodeHandle nh){
        SDL_Init(SDL_INIT_GAMECONTROLLER);
        if(SDL_NumJoysticks()>0){
            controller=SDL_GameControllerOpen(0);
        }
        ackermann_pub=nh.advertise<ackermann_msgs::AckermannDrive>("/ackermann_cmd", 10);
    }

    void gamecontroller_teleop::run(){
        ros::Rate loop_rate(30);

        signal(SIGINT, sigint_handler);

        const float deadzone=0.1f;
        const float ramp=0.15f;
        float cmd_speed=0.0f;

        while(ros::ok() && !g_quit){
            SDL_GameControllerUpdate();

            if(!controller || !SDL_GameControllerGetAttached(controller)){
                if(controller){ SDL_GameControllerClose(controller); controller=nullptr; }
                if(SDL_NumJoysticks()>0) controller=SDL_GameControllerOpen(0);
                if(!controller) ROS_WARN_THROTTLE(5.0, "no game controller found");
            }

            int16_t rawSteer   = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTX);
            int16_t rawSpeed   = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTY);

            speed=-rawSpeed/32768.0f;
            steering=-rawSteer/32768.0f;

            if(std::abs(speed)<deadzone) speed=0.0f;
            if(std::abs(steering)<deadzone) steering=0.0f;

            ackermann_msgs::AckermannDrive msg;

            speed=std::max(std::min(speed,1.0f),-1.0f)*2;
            steering=std::max(std::min(steering,1.0f),-1.0f)*0.34f;

            if(speed > cmd_speed+ramp) cmd_speed+=ramp;
            else if(speed < cmd_speed-ramp) cmd_speed-=ramp;
            else cmd_speed=speed;

            msg.speed = cmd_speed;
            msg.steering_angle = steering;
            ackermann_pub.publish(msg);

            ros::spinOnce();
            loop_rate.sleep();
        }

        ackermann_msgs::AckermannDrive stop;
        ackermann_pub.publish(stop);
        ros::Duration(0.1).sleep();
        if(controller) SDL_GameControllerClose(controller);
        SDL_Quit();
        ros::shutdown();
    }

}
