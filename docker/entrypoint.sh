#!/bin/bash
set -e

Xvfb :1 -screen 0 1600x900x24 &
export DISPLAY=:1
sleep 1

openbox-session &
x11vnc -display :1 -forever -shared -nopw -quiet -xkb &
websockify --web=/usr/share/novnc 6080 localhost:5900 &

source /opt/ros/melodic/setup.bash
source /home/Lab-Car-2/devel/setup.bash

exec "$@"
