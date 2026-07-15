#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <cmath>
#include <ros/ros.h>
#include <string>
#include <sstream> 
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <ackermann_msgs/AckermannDrive.h>

namespace gazebo{
  class mushr_ackermann_plugin: public ModelPlugin{
    
    class PID{
      public:
        double kp; double ki; double kd;
        double prev_error=0;double integral=0;
      PID(): kp(0), ki(0), kd(0){}
      PID(double kp, double ki, double kd){
        this->kp=kp;
        this->ki=ki;
        this->kd=kd;
      }


      double compute(double target, double current, double dt){
        double error=target-current;
        integral+=dt*error;
        double derrivative=(error-prev_error)/dt;
        prev_error=error;
        return kp*error+ki*integral+kd*derrivative;
      }
    };


    private:
      double speed=0;
      double steering_angle=0;
      double wheelbase=0.2965;
      double wheel_separation=0.230;
      double wheel_radius=0.050;
      double max_torque=10.0;
      physics::ModelPtr model;
      physics::JointPtr frontLeft;
      physics::JointPtr frontRight;
      physics::JointPtr backLeft;
      physics::JointPtr backRight;
      physics::JointPtr left_steer;
      physics::JointPtr right_steer;
      common::Time lastUpdateTime;

      ros::Subscriber sub;
      ros::Publisher steer_pub;
      ros::Publisher wheel_vel_pub;
      ros::NodeHandle nodeHandler; 

      event::ConnectionPtr updateConnection;

      PID steering_pid_left;
      PID steering_pid_right;


      


    public:
      void Load(physics::ModelPtr model, sdf::ElementPtr _sdf){

        this->model=model;
        this->nodeHandler=ros::NodeHandle("ackermann_drive_plugin");

        // throttle joint
        this->frontLeft= model->GetJoint(_sdf->Get<std::string>("left_front_joint"));
        this->frontRight= model->GetJoint(_sdf->Get<std::string>("right_front_joint"));
        this->backLeft= model->GetJoint(_sdf->Get<std::string>("left_rear_joint"));
        this->backRight= model->GetJoint(_sdf->Get<std::string>("right_rear_joint"));

        //steering joint
        this->left_steer=model->GetJoint(_sdf->Get<std::string>("left_steering_joint"));
        this->right_steer=model->GetJoint(_sdf->Get<std::string>("right_steering_joint"));

        // car geometry, used for ackermann steering + per-wheel speeds
        if(_sdf->HasElement("wheelbase")) this->wheelbase=_sdf->Get<double>("wheelbase");
        if(_sdf->HasElement("wheel_separation")) this->wheel_separation=_sdf->Get<double>("wheel_separation");
        if(_sdf->HasElement("wheel_radius")) this->wheel_radius=_sdf->Get<double>("wheel_radius");
        if(_sdf->HasElement("max_torque")) this->max_torque=_sdf->Get<double>("max_torque");

        this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&mushr_ackermann_plugin::OnUpdate, this));

        int argc=0;

        if(!ros::isInitialized()){
          ros::init(argc,nullptr, "ackerman_driving_plugin");
        }

        // setting up publisher for the steering 
        this->sub=this->nodeHandler.subscribe<ackermann_msgs::AckermannDrive>("/ackermann_cmd",1,&mushr_ackermann_plugin::onCommand2, this);
        this->steer_pub = this->nodeHandler.advertise<std_msgs::Float64>("/steering_angle",1);

        // setting up the steering pids, one per steering joint so their
        // integral/derivative state doesn't mix
        this->steering_pid_left= PID(_sdf->Get<double>("kp"),_sdf->Get<double>("ki"),_sdf->Get<double>("kd"));
        this->steering_pid_right= PID(_sdf->Get<double>("kp"),_sdf->Get<double>("ki"),_sdf->Get<double>("kd"));
        this->lastUpdateTime=this->model->GetWorld()->SimTime();

        // setting up publisher for wheel velocities
        this->wheel_vel_pub=this->nodeHandler.advertise<std_msgs::Float64MultiArray>("/odom",1);
        
        
        
      }
      void onCommand(const std_msgs::String::ConstPtr &msg){
        std::stringstream ss(msg->data);
        ss >> this->speed>> this->steering_angle;

      }

      
      void onCommand2(const ackermann_msgs::AckermannDrive::ConstPtr& cmd){
        this->steering_angle=cmd->steering_angle;
        this->speed=cmd->speed;
      }

      void OnUpdate(){
        ros::spinOnce();

        // ackermann geometry: in a turn the inner wheels follow a tighter
        // radius than the outer ones, so each wheel needs its own steering
        // angle and rolling speed or they scrub against each other
        double v=this->speed; // m/s at the rear axle center
        double delta=this->steering_angle;
        double target_left=delta, target_right=delta;
        double w_fl, w_fr, w_rl, w_rr; // wheel angular velocities, rad/s
        if(std::fabs(delta)<1e-3){
          w_fl=w_fr=w_rl=w_rr=v/this->wheel_radius;
        }else{
          double R=this->wheelbase/std::tan(delta); // signed turn radius, >0 turning left
          double half=this->wheel_separation/2.0;
          target_left=std::atan(this->wheelbase/(R-half));
          target_right=std::atan(this->wheelbase/(R+half));
          w_rl=v*((R-half)/R)/this->wheel_radius;
          w_rr=v*((R+half)/R)/this->wheel_radius;
          w_fl=v*(std::hypot(this->wheelbase,R-half)/std::fabs(R))/this->wheel_radius;
          w_fr=v*(std::hypot(this->wheelbase,R+half)/std::fabs(R))/this->wheel_radius;
        }

        this->backLeft->SetParam("fmax", 0, this->max_torque);
        this->backLeft->SetParam("vel",0, w_rl);
        this->backRight->SetParam("fmax", 0, this->max_torque);
        this->backRight->SetParam("vel",0, w_rr);
        this->frontLeft->SetParam("fmax", 0, this->max_torque);
        this->frontLeft->SetParam("vel",0, w_fl);
        this->frontRight->SetParam("fmax", 0, this->max_torque);
        this->frontRight->SetParam("vel",0, w_fr);


        common::Time currentTime = this->model->GetWorld()->SimTime();
        double dt=(currentTime - this->lastUpdateTime).Double();
        this->lastUpdateTime = currentTime;
        double calculatedVelocity_left=steering_pid_left.compute(target_left, this->left_steer->Position(0),dt);
        double calculatedVelocity_right=steering_pid_right.compute(target_right, this->right_steer->Position(0),dt);

        this->left_steer->SetParam("fmax", 0, 2.0);
        this->right_steer->SetParam("fmax", 0, 2.0);
        this->left_steer->SetParam("vel",0,calculatedVelocity_left );
        this->right_steer->SetParam("vel",0, calculatedVelocity_right);

        std_msgs::Float64 steer_msg;
        steer_msg.data=this->left_steer->Position(0);
        this->steer_pub.publish(steer_msg);

        std_msgs::Float64MultiArray wheel_vel_msg;
        wheel_vel_msg.data={this->backLeft->GetVelocity(0),this->backRight->GetVelocity(0), this->frontLeft->GetVelocity(0), this->frontRight->GetVelocity(0)};
        this->wheel_vel_pub.publish(wheel_vel_msg);




      }
  };
  GZ_REGISTER_MODEL_PLUGIN(mushr_ackermann_plugin);
}