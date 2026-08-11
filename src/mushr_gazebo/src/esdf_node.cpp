// esdf_node.cpp
// Subscribes to /cloud (PointCloud2 in the laser frame), transforms each scan
// into the map frame via tf, and inserts it into an octree, same as
// mapping_node. On a timer, runs a DynamicEDTOctomap Euclidean distance
// transform over the octree and publishes a colored voxel slice as a
// visualization_msgs/MarkerArray on /esdf_markers, rendered with rviz's
// built-in MarkerArray display. Distance computation is decoupled from cloud
// insertion (recomputing the full distance field is much heavier than an
// octree insert) and only run at ESDF_RATE.

#include <algorithm>
#include <memory>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>
#include <octomap/octomap.h>
#include <dynamicEDT3D/dynamicEDTOctomap.h>
#include <mushr_gazebo/GetDistanceGradient.h>

namespace {
const double ESDF_RATE = 2.0;       // Hz, distance field recompute + publish rate
const float MAX_DIST = 2.0;         // m, distances beyond this are clamped
const double SLICE_Z = 0.275;       // m, height (in map frame) of the visualized slice
}

class EsdfMapper {
  ros::NodeHandle private_nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::Timer timer_;
  ros::ServiceServer dist_srv_;
  tf::TransformListener tf_;
  octomap::OcTree tree_;
  std::string map_frame_;

  // Most recent distance field, retained between timer cycles so the service
  // can query it. edt_min_/edt_max_ are the xy bounds it was computed over.
  std::unique_ptr<DynamicEDTOctomap> edt_;
  octomap::point3d edt_min_;
  octomap::point3d edt_max_;

public:
  EsdfMapper(ros::NodeHandle& nh) : private_nh_("~"), tree_(0.05), map_frame_("map") {
    sub_ = nh.subscribe("/cloud", 5, &EsdfMapper::cloudCb, this);
    pub_ = nh.advertise<visualization_msgs::MarkerArray>("/esdf_markers", 1, true);
    timer_ = nh.createTimer(ros::Duration(1.0 / ESDF_RATE), &EsdfMapper::timerCb, this);
    dist_srv_ = private_nh_.advertiseService("get_distance_gradient", &EsdfMapper::distanceGradientCb, this);
  }

  void cloudCb(const sensor_msgs::PointCloud2::ConstPtr& msg) {
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
  }

  void timerCb(const ros::TimerEvent&) {
    if (tree_.size() == 0) return;

    double minX, minY, minZ, maxX, maxY, maxZ;
    tree_.getMetricMin(minX, minY, minZ);
    tree_.getMetricMax(maxX, maxY, maxZ);
    double res = tree_.getResolution();
    octomap::point3d bbxMin(minX, minY, SLICE_Z - res);
    octomap::point3d bbxMax(maxX, maxY, SLICE_Z + res);

    edt_.reset(new DynamicEDTOctomap(MAX_DIST, &tree_, bbxMin, bbxMax, /*treatUnknownAsOccupied=*/false));
    edt_->update();
    edt_min_ = bbxMin;
    edt_max_ = bbxMax;

    publishMarkers(*edt_, bbxMin, bbxMax);
  }

  bool distanceGradientCb(mushr_gazebo::GetDistanceGradient::Request& req,
                          mushr_gazebo::GetDistanceGradient::Response& res) {
    res.success = false;
    res.distance = 0.0;
    res.gradient_x = 0.0;
    res.gradient_y = 0.0;

    if (!edt_) return true;  // no map data yet

    if (req.x < edt_min_.x() || req.x > edt_max_.x() ||
        req.y < edt_min_.y() || req.y > edt_max_.y()) {
      return true;  // query point outside the current distance field
    }

    const float eps = 0.05f;
    float dc = edt_->getDistance(octomap::point3d(req.x, req.y, SLICE_Z));
    float dxp = edt_->getDistance(octomap::point3d(req.x + eps, req.y, SLICE_Z));
    float dxm = edt_->getDistance(octomap::point3d(req.x - eps, req.y, SLICE_Z));
    float dyp = edt_->getDistance(octomap::point3d(req.x, req.y + eps, SLICE_Z));
    float dym = edt_->getDistance(octomap::point3d(req.x, req.y - eps, SLICE_Z));

    if (dc == DynamicEDTOctomap::distanceValue_Error ||
        dxp == DynamicEDTOctomap::distanceValue_Error ||
        dxm == DynamicEDTOctomap::distanceValue_Error ||
        dyp == DynamicEDTOctomap::distanceValue_Error ||
        dym == DynamicEDTOctomap::distanceValue_Error) {
      return true;  // query or a finite-difference neighbor fell outside the field
    }

    res.distance = dc;
    res.gradient_x = (dxp - dxm) / (2.0f * eps);
    res.gradient_y = (dyp - dym) / (2.0f * eps);
    res.success = true;
    return true;
  }

  void publishMarkers(DynamicEDTOctomap& edt, const octomap::point3d& bbxMin,
                       const octomap::point3d& bbxMax) {
    double res = tree_.getResolution();

    visualization_msgs::MarkerArray arr;
    visualization_msgs::Marker m;
    m.header.frame_id = map_frame_;
    m.header.stamp = ros::Time::now();
    m.ns = "esdf";
    m.id = 0;
    m.type = visualization_msgs::Marker::CUBE_LIST;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = m.scale.y = m.scale.z = res;
    m.pose.orientation.w = 1.0;

    for (double x = bbxMin.x(); x <= bbxMax.x(); x += res) {
      for (double y = bbxMin.y(); y <= bbxMax.y(); y += res) {
        octomap::point3d p(x, y, SLICE_Z);
        float d = edt.getDistance(p);
        if (d == DynamicEDTOctomap::distanceValue_Error) continue;

        geometry_msgs::Point pt;
        pt.x = x; pt.y = y; pt.z = SLICE_Z;
        m.points.push_back(pt);

        // Red (at an obstacle) -> green (>= MAX_DIST away).
        float t = std::min(std::max(d / MAX_DIST, 0.0f), 1.0f);
        std_msgs::ColorRGBA c;
        c.r = 1.0 - t;
        c.g = t;
        c.b = 0.0;
        c.a = 0.6;
        m.colors.push_back(c);
      }
    }

    arr.markers.push_back(m);
    pub_.publish(arr);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "esdf_node");
  ros::NodeHandle nh;
  EsdfMapper node(nh);
  ros::spin();
}
