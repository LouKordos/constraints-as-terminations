cmake_minimum_required(VERSION 3.22)

# Silence FetchContent CMP0135 warning for CMake 3.24+
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(cat_controller)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(cat_perception_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)
find_package(ament_index_cpp)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)
find_package(cppzmq REQUIRED)

add_library(inference_engine STATIC src/inference_engine.cpp)
target_include_directories(inference_engine PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>) # Include the current package include dir
target_link_libraries(inference_engine PUBLIC ${TORCH_LIBRARIES})

add_executable(cat_controller src/cat_control_node.cpp src/motor_crc.cpp src/low_level_mode_enabler.cpp)
target_include_directories(cat_controller PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>) # Include the current package include dir
target_link_libraries(cat_controller PUBLIC 
  inference_engine 
  rclcpp::rclcpp 
  tf2::tf2 
  ament_index_cpp::ament_index_cpp 
  ${std_msgs_TARGETS}
  ${unitree_go_TARGETS}
  ${unitree_api_TARGETS}
  ${cat_perception_msgs_TARGETS}
  fmt::fmt
  nlohmann_json::nlohmann_json
  cppzmq # TODO: Remove unused dependencies after migration to new processor
)

add_executable(elevation_map_processing_node src/elevation_map_processing_node.cpp)
target_include_directories(elevation_map_processing_node PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>) # Include the current package include dir
target_link_libraries(
  elevation_map_processing_node
  rclcpp::rclcpp
  ament_index_cpp::ament_index_cpp
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${grid_map_msgs_TARGETS}
  ${cat_perception_msgs_TARGETS}
  fmt::fmt
  nlohmann_json::nlohmann_json
)

add_executable(projected_gravity_equivalence_test_node src/projected_gravity_equivalence_test_node.cpp)
target_link_libraries(projected_gravity_equivalence_test_node PUBLIC rclcpp::rclcpp tf2::tf2)

# Isolated helper binary for entering low level control mode because ROS approach is not functional
find_package(unitree_sdk2 REQUIRED)
find_package(Threads REQUIRED)
add_executable(release_motion_mode src/release_motion_mode.cpp)
target_link_libraries(release_motion_mode PRIVATE unitree_sdk2 Threads::Threads)

# This makes ros2 run find the executable when the ws is sourced
install(TARGETS
  cat_controller
  elevation_map_processing_node
  projected_gravity_equivalence_test_node
  release_motion_mode
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()