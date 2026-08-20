FROM osrf/ros:melodic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Minimal virtual desktop + browser-based VNC stack, so Gazebo's GUI is
# reachable at http://localhost:6080 without installing an X server on
# the host (XQuartz/VcXsrv are the usual friction point on Mac/Windows).
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        x11vnc \
        novnc \
        websockify \
        openbox \
        python-rosdep \
        ros-melodic-joy \
        ros-melodic-joint-state-publisher \
        ros-melodic-robot-state-publisher \
        libsdl2-dev \
        liboctomap-dev \
        libdynamicedt3d-dev \
        ros-melodic-serial \
    && rm -rf /var/lib/apt/lists/*

ENV CATKIN_WS=/home/Lab-Car-2
WORKDIR $CATKIN_WS

COPY src $CATKIN_WS/src

RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && \
    catkin_init_workspace src && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y --skip-keys='pedsim_gazebo_plugin pedsim_simulator' || true && \
    catkin_make"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["roslaunch", "mushr_gazebo", "gazebo-sim.launch"]
