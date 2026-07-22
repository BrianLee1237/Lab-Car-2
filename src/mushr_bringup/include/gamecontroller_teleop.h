#ifndef GAMECONTROLLER_TELEOP
#define GAMECONTROLLER_TELEOP

#include <ros/ros.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <stdlib.h>
#include <SDL2/SDL.h>

namespace GameControllerTeleop{

    class gamecontroller_teleop{

        public:
            gamecontroller_teleop(ros::NodeHandle nh);
            void run();

        private:
            ros::Publisher ackermann_pub;
            float speed=0.0;           
            float steering=0.0;
    
            SDL_GameController* controller = nullptr;
    };
}


#endif