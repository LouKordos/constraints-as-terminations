#!/bin/bash
set -e pipefail

source /opt/ros/$ROS_DISTRO/setup.bash
set -x # After verbose source

BASE_DIR=/app # In case you want to run outside docker

# For alternative odom, this will conflict with go2_bringup so do not run both at the time!:
mkdir -p $BASE_DIR/odom_alternative_ws/src/third_party/
cd $BASE_DIR/odom_alternative_ws/src/third_party
git clone https://github.com/inria-paris-robotics-lab/go2_odometry.git || true # Do not pull due to patching below: (cd go2_odometry && git pull)
# Lower foot contact threshold to finish initialization correctly (OBSOLETE DUE TO UPSTREAM CHANGES)
# sed -i "s|if np.min(f_contact) > 30|if np.min(f_contact) > 20|" ./go2_odometry/scripts/feet_to_odom_inekf.py
git clone https://github.com/inria-paris-robotics-lab/go2_description.git || (cd go2_description && git pull)
git clone https://github.com/Unitree-Go2-Robot/unitree_go.git || (cd unitree_go && git pull)
# git clone https://github.com/LouKordos/elevation_mapping_cupy.git -b ros2_humble || (cd elevation_mapping_cupy && git pull)

mkdir -p $BASE_DIR/ros2_ws/src/third_party # Use third_party for any external packages because it's ignored by git
cd $BASE_DIR/ros2_ws/src/third_party # For custom code, use ros2_ws/src/

# For /tf topic and improved Go2 ROS2 integration, Odom
# TODO: Use dependencies.repos and vcs, pin commits
git clone -b humble https://github.com/Unitree-Go2-Robot/go2_robot.git go2_robot || (cd go2_robot && git pull)
sed -i 's|https://github.com/Unitree-Go2-Robot/go2_driver.git|https://github.com/LouKordos/go2_driver.git|' go2_robot/dependencies.repos # Replacement with fork only needed until PR is merged
vcs import < go2_robot/dependencies.repos
git clone https://github.com/Ericsii/livox_ros_driver2 -b feature/use-standard-unit || (cd livox_ros_driver2 && git pull)
git clone --recurse-submodules https://github.com/LouKordos/FAST_LIO_ROS2.git || (cd FAST_LIO_ROS2 && git pull)
git clone https://github.com/LouKordos/LiDAR_IMU_Init.git/ || (cd LiDAR_IMU_Init && git pull)
git clone --recurse-submodules https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0.git HesaiLidar_ROS_2.0 || (cd HesaiLidar_ROS_2.0 && git pull)
# git clone https://github.com/unitreerobotics/unitree_ros2 || (cd unitree_ros2 && git pull) # ROS2 integration provided by go2_robot.git

cd $BASE_DIR/ros2_ws
ROSDEP_MARKER=/rosdep-bootstrap-ros-ws.marker
if [[ ! -f "${ROSDEP_MARKER}" ]]; then
    echo "${ROSDEP_MARKER} missing, initializing and updating rosdep..."
    apt-get install -y libyaml-cpp-dev libboost-all-dev ros-$ROS_DISTRO-realsense2-camera ros-$ROS_DISTRO-pointcloud-to-laserscan
    rosdep init || true
    rosdep update
    touch "${ROSDEP_MARKER}"
    echo "Rosdep marker created at ${ROSDEP_MARKER}"
else
    echo "Found marker at ${ROSDEP_MARKER}, skipping rosdep init and update."
fi

rosdep install --from-paths $BASE_DIR/ros2_ws/src --ignore-src -r -y
rosdep install --from-paths $BASE_DIR/odom_alternative_ws/src --ignore-src -r -y

export CMAKE_EXPORT_COMPILE_COMMANDS=ON
COLCON_ARGS=(
    --cmake-args
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_TESTING=OFF"
    "-DPYTHON_EXECUTABLE=$(which python3)"
    "-DCMAKE_CXX_FLAGS=-Wl,--allow-shlib-undefined -Wall -Wextra -Wpedantic -Wshadow"
    --packages-skip convex_plane_decomposition convex_plane_decomposition_ros
    --parallel-workers $(nproc)
)
FIRST_BUILD_MARKER=/colcon-ros2_ws_clean_build.marker

if [[ -d "${BASE_DIR}/odom_alternative_ws" ]]; then
    if [[ ! -f "${FIRST_BUILD_MARKER}" ]]; then
        echo "First script run inside docker, cleaning cmake cache and installing marker..."
        rm -rf $BASE_DIR/odom_alternative_ws/{build,install,log}
        cd $BASE_DIR/odom_alternative_ws
        colcon build --cmake-clean-cache "${COLCON_ARGS[@]}"
    else
        colcon build "${COLCON_ARGS[@]}"
    fi
fi

cd $BASE_DIR/ros2_ws

if [[ ! -f "${FIRST_BUILD_MARKER}" ]]; then
    echo "First script run inside docker, cleaning cmake cache and installing marker..."
    rm -rf $BASE_DIR/ros2_ws/{build,install,log}
    colcon build --cmake-clean-cache "${COLCON_ARGS[@]}"
    touch "${FIRST_BUILD_MARKER}" # Only if build succeeded
    echo "First build marker created at ${FIRST_BUILD_MARKER}"
else
    colcon build "${COLCON_ARGS[@]}"
fi

chmod -R a+rwX $BASE_DIR/ros2_ws $BASE_DIR/odom_alternative_ws

echo "ROS2 workspace bootstrap finished."
