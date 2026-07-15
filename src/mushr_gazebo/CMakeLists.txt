cmake_minimum_required(VERSION 3.0.2)
project(mushr_gazebo)

find_package(catkin REQUIRED COMPONENTS
  ackermann_msgs
  gazebo_msgs
  gazebo_ros
  roscpp
  rospy
)

find_package(gazebo REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES mushr_gazebo
 CATKIN_DEPENDS ackermann_msgs gazebo_msgs gazebo_ros roscpp rospy
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)


add_library(mushr_gazebo plugins/mushr_ackermann_plugin.cpp)
add_dependencies(mushr_gazebo ${catkin_EXPORTED_TARGETS})
target_link_libraries(mushr_gazebo ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_executable(gazebo_keyboard_teleop src/keyboard_teleop.cpp src/keyboard_teleop_node.cpp)
add_dependencies(gazebo_keyboard_teleop ${catkin_EXPORTED_TARGETS})
target_link_libraries(gazebo_keyboard_teleop
  ${catkin_LIBRARIES}
)
if(APPLE)
  # CGEventSourceKeyState, used to poll physical key state in the teleop
  target_link_libraries(gazebo_keyboard_teleop "-framework CoreGraphics")
endif()

