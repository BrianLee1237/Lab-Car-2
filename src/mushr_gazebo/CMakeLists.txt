cmake_minimum_required(VERSION 3.0.2)
project(mushr_gazebo)

find_package(catkin REQUIRED COMPONENTS
  ackermann_msgs
  gazebo_msgs
  gazebo_ros
  roscpp
  rospy
  laser_geometry
  octomap_msgs
  visualization_msgs
)

find_package(Eigen3 REQUIRED)

find_package(gazebo REQUIRED)

find_package(octomap REQUIRED)
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
  ${EIGEN3_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)


add_library(mushr_gazebo plugins/mushr_ackermann_plugin.cpp)
add_dependencies(mushr_gazebo ${catkin_EXPORTED_TARGETS})
target_link_libraries(mushr_gazebo ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_executable(gazebo_keyboard_teleop src/keyboard_teleop.cpp src/keyboard_teleop_node.cpp)
add_dependencies(gazebo_keyboard_teleop ${catkin_EXPORTED_TARGETS})
target_link_libraries(gazebo_keyboard_teleop
  ${catkin_LIBRARIES}
)

add_executable(lidar_conversion src/convert_lidar_format.cpp)
add_dependencies(lidar_conversion ${catkin_EXPORTED_TARGETS})
target_link_libraries(lidar_conversion ${catkin_LIBRARIES} Eigen3::Eigen )

add_executable(mapping_node src/mapping_node.cpp)
add_dependencies(mapping_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(mapping_node ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES})




if(APPLE)
  # CGEventSourceKeyState, used to poll physical key state in the teleop
  target_link_libraries(gazebo_keyboard_teleop "-framework CoreGraphics")
endif()

