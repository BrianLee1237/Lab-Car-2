#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose2D
from tf.transformations import euler_from_quaternion

ROBOT_NAME = "myrobot2"   

class ground_truth:
    def __init__(self):
        rospy.init_node('ground_truth')
        self.pub = rospy.Publisher('/ground_truth_pose', Pose2D, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb)
    
    def cb(self, msg):
        if ROBOT_NAME not in msg.name:
            return
        i = msg.name.index(ROBOT_NAME)
        pose = msg.pose[i]
        
        q = pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        out = Pose2D()
        out.x = pose.position.x
        out.y = pose.position.y
        out.theta = yaw
        self.pub.publish(out)

if __name__ == "__main__":
    ground_truth()
    rospy.spin()