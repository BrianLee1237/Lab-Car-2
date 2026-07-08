#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <string>
#include <sstream> 
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>

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

      PID steering_pid;


      


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

        this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&mushr_ackermann_plugin::OnUpdate, this));

        int argc=0;

        if(!ros::isInitialized()){
          ros::init(argc,nullptr, "ackerman_driving_plugin");
        }

        // setting up publisher for the steering 
        this->sub=this->nodeHandler.subscribe<std_msgs::String>("/ackermann_cmd",1,&mushr_ackermann_plugin::onCommand, this);
        this->steer_pub = this->nodeHandler.advertise<std_msgs::Float64>("/steering_angle",1);

        // setting up the steering pid
        this->steering_pid= PID(_sdf->Get<double>("kp"),_sdf->Get<double>("ki"),_sdf->Get<double>("kd"));
        this->lastUpdateTime=this->model->GetWorld()->SimTime();

        // setting up publisher for wheel velocities
        this->wheel_vel_pub=this->nodeHandler.advertise<std_msgs::Float64MultiArray>("/odom",1);
        
        
        
      }
      void onCommand(const std_msgs::String::ConstPtr &msg){
        std::stringstream ss(msg->data);
        ss >> this->speed>> this->steering_angle;

      }
      void OnUpdate(){
        ros::spinOnce();


        this->backLeft->SetParam("fmax", 0, 2.0);
        this->backLeft->SetParam("vel",0, this->speed);
        this->backRight->SetParam("fmax", 0, 2.0);
        this->backRight->SetParam("vel",0, this->speed);
        this->frontLeft->SetParam("fmax", 0, 2.0);
        this->frontLeft->SetParam("vel",0, this->speed);
        this->frontRight->SetParam("fmax", 0, 2.0);
        this->frontRight->SetParam("vel",0, this->speed);


        common::Time currentTime = this->model->GetWorld()->SimTime();
        double dt=(currentTime - this->lastUpdateTime).Double();
        this->lastUpdateTime = currentTime;
        double calculatedVelocity_left=steering_pid.compute(this->steering_angle, this->left_steer->Position(0),dt);
        double calculatedVelocity_right=steering_pid.compute(this->steering_angle, this->right_steer->Position(0),dt);

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