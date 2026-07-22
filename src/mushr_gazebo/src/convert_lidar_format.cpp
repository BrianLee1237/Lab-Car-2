#include <ros/ros.h>
#include <laser_geometry/laser_geometry.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>

class Scan2Cloud {
  ros::Subscriber sub_;
  ros::Publisher pub_;
  laser_geometry::LaserProjection projector_;
public:
  Scan2Cloud(ros::NodeHandle& nh) {
    sub_ = nh.subscribe("/scan", 10, &Scan2Cloud::cb, this);
    pub_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud", 10);
  }
  void cb(const sensor_msgs::LaserScan::ConstPtr& scan) { // callback for conversion from the scan format into the pointcloud2 format
    sensor_msgs::PointCloud2 cloud;
    projector_.projectLaser(*scan, cloud);   // stays in "laser" frame
    pub_.publish(cloud);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "scan_to_cloud");
  ros::NodeHandle nh;
  Scan2Cloud node(nh);
  ros::spin();
}