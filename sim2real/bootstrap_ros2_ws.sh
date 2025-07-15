#!/bin/bash
set -e pipefail

source /opt/ros/$ROS_DISTRO/setup.bash
set -x # After verbose source

BASE_DIR=/app # In case you want to run outside docker

# For alternative odom, this will conflict with go2_bringup so do not run both at the time!:
mkdir -p $BASE_DIR/odom_alternative_ws/src
sudo chmod -R 777 $BASE_DIR/odom_alternative_ws
cd $BASE_DIR/odom_alternative_ws/src
git clone https://github.com/inria-paris-robotics-lab/go2_odometry.git
git clone https://github.com/inria-paris-robotics-lab/go2_description.git
git clone https://github.com/Unitree-Go2-Robot/unitree_go.git

mkdir -p $BASE_DIR/ros2_ws/src/third_party # Use third_party for any external packages because it's ignored by git
sudo chmod -R 777 $BASE_DIR/ros2_ws
cd $BASE_DIR/ros2_ws/src/third_party # For custom code, use ros2_ws/src/

# For /tf topic and improved Go2 ROS2 integration
git clone -b humble https://github.com/Unitree-Go2-Robot/go2_robot.git go2_robot || (cd go2_robot && git pull)
vcs import < go2_robot/dependencies.repos
git clone --recurse-submodules https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0.git HesaiLidar_ROS_2.0 || (cd HesaiLidar_ROS_2.0 && git pull)
# git clone https://github.com/unitreerobotics/unitree_ros2 || (cd unitree_ros2 && git pull) # Provided by go2_robot.git

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

rosdep install --from-paths src --ignore-src -r -y

if [[ -d "${BASE_DIR}/odom_alternative_ws" ]]; then
    rosdep install --from-paths $BASE_DIR/odom_alternative_ws/src --ignore-src -r -y
    cd $BASE_DIR/odom_alternative_ws
    colcon build
    cd $BASE_DIR/ros2_ws
fi

FIRST_BUILD_MARKER=/colcon-ros2_ws_clean_build.marker
COLCON_ARGS=() # To prevent duplication
if [[ ! -f "${FIRST_BUILD_MARKER}" ]]; then
    echo "First script run inside docker, cleaning cmake cache and installing marker..."
    rm -rf $BASE_DIR/ros2_ws/{build,install,log}
    colcon build --cmake-clean-cache "${COLCON_ARGS[@]}"
    touch "${FIRST_BUILD_MARKER}" # Only if build succeeded
    echo "First build marker created at ${FIRST_BUILD_MARKER}"
else
    colcon build "${COLCON_ARGS[@]}"
fi

echo "ROS2 workspace bootstrap finished."