cmake_minimum_required(VERSION 3.8)
project(cat_controller)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)

add_executable(cat_controller src/cat_control_node.cpp src/motor_crc.cpp src/low_level_mode_enabler.cpp)
target_include_directories(cat_controller PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>) # Include the current package include dir

target_link_libraries(cat_controller PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS} ${unitree_go_TARGETS} ${unitree_api_TARGETS} ${TORCH_LIBRARIES} fmt::fmt)

# This makes ros2 run find the executable when the ws is sourced
install(TARGETS
  cat_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()