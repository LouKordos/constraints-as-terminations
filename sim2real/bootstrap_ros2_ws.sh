#!/bin/bash
set -e pipefail

source /opt/ros/$ROS_DISTRO/setup.bash
set -x # After verbose source

BASE_DIR=/app/sim2real/ # In case you want to run outside docker

# TODO: Use dependencies.repos and vcs, pin commits
# Use third_party for any external packages because it's ignored by git. For custom code, use ros2_ws directly as that will be tracked by git.
mkdir -p $BASE_DIR/ros2_ws/src/third_party && cd $BASE_DIR/ros2_ws/src/third_party
git clone https://github.com/inria-paris-robotics-lab/go2_odometry.git || (cd go2_odometry && git pull)
git clone https://github.com/inria-paris-robotics-lab/unitree_description.git || (cd unitree_description && git pull)
git clone https://github.com/Ericsii/livox_ros_driver2 -b feature/use-standard-unit || (cd livox_ros_driver2 && git pull)
git clone --recurse-submodules https://github.com/LouKordos/FAST_LIO_ROS2.git || (cd FAST_LIO_ROS2 && git pull)
# git clone https://github.com/LouKordos/LiDAR_IMU_Init.git/ || (cd LiDAR_IMU_Init && git pull)
git clone https://github.com/unitreerobotics/unitree_ros2 || (cd unitree_ros2 && git pull)

#  Copied from go2 repo because its needed for sending valid motor commands and they do not install these header files automatically
cp $BASE_DIR/ros2_ws/src/third_party/unitree_ros2/example/src/include/common/motor_crc.h $BASE_DIR/ros2_ws/src/cat_controller/include/cat_controller/motor_crc.h
cp $BASE_DIR/ros2_ws/src/third_party/unitree_ros2/example/src/src/common/motor_crc.cpp $BASE_DIR/ros2_ws/src/cat_controller/src/motor_crc.cpp
sed -i 's|#include "motor_crc.h"|#include "cat_controller/motor_crc.h"|' "$BASE_DIR/ros2_ws/src/cat_controller/src/motor_crc.cpp"

cd $BASE_DIR/ros2_ws
ROSDEP_MARKER=/rosdep-bootstrap-ros-ws.marker
if [[ ! -f "${ROSDEP_MARKER}" ]]; then
    echo "${ROSDEP_MARKER} missing, initializing and updating rosdep..."
    apt-get update -y
    apt-get install -y jq libyaml-cpp-dev libboost-all-dev ros-$ROS_DISTRO-realsense2-camera ros-$ROS_DISTRO-pointcloud-to-laserscan
    rosdep init || true
    rosdep update
    touch "${ROSDEP_MARKER}"
    echo "Rosdep marker created at ${ROSDEP_MARKER}"
else
    echo "Found marker at ${ROSDEP_MARKER}, skipping rosdep init and update."
fi

rosdep install --from-paths $BASE_DIR/ros2_ws/src --ignore-src -r -y

export CMAKE_EXPORT_COMPILE_COMMANDS=ON
COLCON_ARGS=(
    --cmake-args
    "-G Ninja"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    "-DBUILD_TESTING=OFF"
    "-DPYTHON_EXECUTABLE=$(which python3)"
    "-DCMAKE_CXX_FLAGS=-Wall -Wextra -Wpedantic -Wshadow"
    --parallel-workers $(nproc)
)

FIRST_BUILD_MARKER=/colcon-ros2_ws_clean_build.marker
cd $BASE_DIR/ros2_ws
if [[ ! -f "${FIRST_BUILD_MARKER}" ]]; then
    echo "First script run inside docker, cleaning cmake cache and installing marker..."
    rm -rf ${BASE_DIR}/ros2_ws/{build,install,log}
    colcon build --cmake-clean-cache "${COLCON_ARGS[@]}"
    touch "${FIRST_BUILD_MARKER}" # Only if build succeeded
    echo "First build marker created at ${FIRST_BUILD_MARKER}"
else
    colcon build "${COLCON_ARGS[@]}"
fi

echo "Merging compile_commands.json files for IntelliSense..." # colcon produces a separate compile_commands.json for each package
jq -s 'add' $BASE_DIR/ros2_ws/build/*/compile_commands.json > $BASE_DIR/ros2_ws/build/compile_commands.json

chmod -R a+rwX $BASE_DIR/ros2_ws
echo "ROS2 workspace bootstrap finished."
