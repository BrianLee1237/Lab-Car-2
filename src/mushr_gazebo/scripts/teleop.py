#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from pynput import keyboard
import numpy as np

pressed = set()

def on_press(key):
    try:
        pressed.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed.discard(key.char)
    except AttributeError:
        pass

def compute_cmd():
    speed = 0.0
    steer = 0.0
    
    # forward/back
    if 'w' in pressed:
        speed += 10.0
    if 's' in pressed:
        speed -= 10.0
    
    # steering
    if 'a' in pressed:
        steer += 0.3
    if 'd' in pressed:
        steer -= 0.3
    
    return speed, steer

if __name__ == "__main__":
    rospy.init_node('teleop')
    pub = rospy.Publisher('/ackermann_cmd', String, queue_size=1)
    rospy.sleep(1.0)
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    print("Hold WASD to drive, release to stop. Ctrl-C to quit.")
    rate = rospy.Rate(20)
    current_speed = 0.0
    current_steer = 0.0
    SPEED_RAMP = 50.0   # m/s² ish
    STEER_RAMP = 2.0    # rad/s

    while not rospy.is_shutdown():
        target_speed, target_steer = compute_cmd()
        dt = 1.0 / 20.0
        
        # ramp toward target
        current_speed += np.clip(target_speed - current_speed, -SPEED_RAMP*dt, SPEED_RAMP*dt)
        current_steer += np.clip(target_steer - current_steer, -STEER_RAMP*dt, STEER_RAMP*dt)
        
        pub.publish(String(data=f"{current_speed} {current_steer}"))
        rate.sleep()