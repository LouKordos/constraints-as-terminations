cmake_minimum_required(VERSION 3.8)
project(cat_controller)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Generate json file for VS Code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(cat_controller src/cat_control_node.cpp)
target_link_libraries(cat_controller PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS})

# This makes ros2 run find the executable when the ws is sourced
install(TARGETS
  cat_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()