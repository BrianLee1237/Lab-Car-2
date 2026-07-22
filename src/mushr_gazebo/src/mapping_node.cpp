// mapping_node.cpp
// Subscribes to /cloud (PointCloud2 in the laser frame), transforms each scan
// into the map frame via tf, inserts it into an octree, and publishes the
// occupied voxels as a visualization_msgs/MarkerArray on /octomap_markers,
// rendered with rviz's built-in MarkerArray display.

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <octomap/octomap.h>

class Mapper {
  ros::Subscriber sub_;
  ros::Publisher pub_;
  tf::TransformListener tf_;
  octomap::OcTree tree_;
  std::string map_frame_;

public:
  Mapper(ros::NodeHandle& nh) : tree_(0.05), map_frame_("map") {
    sub_ = nh.subscribe("/cloud", 5, &Mapper::cb, this);
    pub_ = nh.advertise<visualization_msgs::MarkerArray>("/octomap_markers", 1, true);
  }

  void cb(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    tf::StampedTransform T;
    try {
      tf_.waitForTransform(map_frame_, msg->header.frame_id,
                           msg->header.stamp, ros::Duration(0.1));
      tf_.lookupTransform(map_frame_, msg->header.frame_id,
                          msg->header.stamp, T);
    } catch (tf::TransformException& e) {
      ROS_WARN_THROTTLE(1.0, "tf lookup failed: %s", e.what());
      return;
    }

    octomap::point3d origin(T.getOrigin().x(), T.getOrigin().y(), T.getOrigin().z());

    octomap::Pointcloud pc;
    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y"), iz(*msg, "z");
    for (; ix != ix.end(); ++ix, ++iy, ++iz) {
      if (!std::isfinite(*ix)) continue;
      tf::Vector3 pm = T * tf::Vector3(*ix, *iy, *iz);
      pc.push_back(pm.x(), pm.y(), pm.z());
    }

    tree_.insertPointCloud(pc, origin);
    publishMarkers();
  }

  void publishMarkers() {
    visualization_msgs::MarkerArray arr;
    visualization_msgs::Marker m;
    m.header.frame_id = map_frame_;
    m.header.stamp = ros::Time::now();
    m.ns = "octomap";
    m.id = 0;
    m.type = visualization_msgs::Marker::CUBE_LIST;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = m.scale.y = m.scale.z = tree_.getResolution();
    m.color.r = 0.1; m.color.g = 0.6; m.color.b = 1.0; m.color.a = 1.0;
    m.pose.orientation.w = 1.0;

    for (auto it = tree_.begin_leafs(), end = tree_.end_leafs(); it != end; ++it) {
      if (!tree_.isNodeOccupied(*it)) continue;
      geometry_msgs::Point p;
      p.x = it.getX(); p.y = it.getY(); p.z = it.getZ();
      m.points.push_back(p);
    }

    arr.markers.push_back(m);
    pub_.publish(arr);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "mapping_node");
  ros::NodeHandle nh;
  Mapper node(nh);
  ros::spin();
}