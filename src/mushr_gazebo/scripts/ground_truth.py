#!/usr/bin/env python3
# Publishes gazebo ground-truth pose as /ground_truth_pose (Pose2D) and, more
# importantly, broadcasts tf map -> base_footprint so the tf tree has a "map"
# frame for mapping_node / rviz.
#
# NOTE: deliberately avoids `tf2_ros` and `tf.transformations` -- those import
# native PyKDL/tf2 bindings that segfault (exit -11) in this conda ROS env.
# We publish the transform directly to /tf (exactly what a broadcaster does)
# and compute the yaw quaternion with plain math.
import math
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose2D, TransformStamped
from tf2_msgs.msg import TFMessage

ROBOT_NAME = "mushr"           # preferred gazebo model name (-model mushr)
MAP_FRAME = "map"
BASE_FRAME = "base_footprint"  # URDF root link; robot_state_publisher chains this to laser

# Fallback keywords if ROBOT_NAME isn't found exactly (case-insensitive substring).
ROBOT_KEYWORDS = ("mushr", "racecar", "robot")


def yaw_from_quat(q):
    # z-axis yaw from a quaternion (pure python; no tf.transformations)
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ground_truth:
    def __init__(self):
        rospy.init_node('ground_truth')
        self.pub = rospy.Publisher('/ground_truth_pose', Pose2D, queue_size=1)
        self.tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=1)
        self._logged = False
        rospy.loginfo("ground_truth started; waiting for /gazebo/model_states ...")
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb)

    def pick_robot(self, names):
        """Return the index of the robot model, or None."""
        if ROBOT_NAME in names:
            return names.index(ROBOT_NAME)
        for idx, n in enumerate(names):
            low = n.lower()
            if any(k in low for k in ROBOT_KEYWORDS):
                return idx
        return None

    def cb(self, msg):
        i = self.pick_robot(msg.name)
        # Log what we see exactly once, so the terminal shows the real model names.
        if not self._logged:
            self._logged = True
            chosen = msg.name[i] if i is not None else "NONE MATCHED"
            rospy.loginfo("model_states names=%s ; using robot='%s'",
                          list(msg.name), chosen)
        if i is None:
            rospy.logwarn_throttle(
                5.0, "No robot model matched %r or %r in %s -- set ROBOT_NAME.",
                ROBOT_NAME, ROBOT_KEYWORDS, list(msg.name))
            return

        pose = msg.pose[i]
        yaw = yaw_from_quat(pose.orientation)

        out = Pose2D()
        out.x = pose.position.x
        out.y = pose.position.y
        out.theta = yaw
        self.pub.publish(out)

        # Broadcast map -> base_footprint (ground-projected: z=0, roll=pitch=0).
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = MAP_FRAME
        t.child_frame_id = BASE_FRAME
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_pub.publish(TFMessage([t]))


if __name__ == "__main__":
    ground_truth()
    rospy.spin()
