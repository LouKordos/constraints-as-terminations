# Go2 sim-to-real deployment

The LoComposition deployment stack provides the same observation and control interfaces used in simulation. It separates map generation, map post-processing, policy inference, state estimation, and robot communication so that each component can be checked before locomotion is enabled.

## Architecture

- `locomposition_controller` runs the traced TorchScript policy at 50 Hz and republishes the latest PD target from a 500 Hz low-level loop.
- `locomposition_perception_msgs` defines the processed 13×11 robot-centric elevation-map message.
- `locomposition_state_estimation` launches the configured odometry and Livox inputs.
- `locomposition_bringup` starts controller, state estimation, static transforms, map processing, and visualization settings.
- `elevation_mapping_cupy` runs separately and publishes the global elevation map consumed by the processing node.

The processing node runs faster than policy inference to avoid outdated elevation maps. It samples the latest global map using the latest transform, computes heights relative to the base, and produces the same yaw-aligned 8 cm grid used during training.

## Prerequisites

You need a Unitree Go2 with low-level control access, a Linux host or onboard computer with Docker, ROS 2 networking to the robot, and a Livox MID-360 for the perceptive policy. The reported experiments used Vicon pose input to remove state-estimation drift as a confound, but the controller itself can use a compatible onboard odometry source instead.

## Synchronize clocks first

Accurate time is essential for map lookup, TF, and state freshness. Set up one machine that has reasonably accurate time as the time server so that you get low jitter:

1. Add `allow 192.168.123.0/24` to `/etc/chrony/chrony.conf` on the workstation and allow NTP through its firewall.
2. Add `server WORKSTATION_IP iburst prefer minpoll 0 maxpoll` to `/etc/chrony/chrony.conf` on the Go2.
3. Restart chrony and inspect `chronyc tracking` on both machines.

This should achieve <1ms offset, ideally lower.

## Robot and DDS network

Follow Unitree's network setup and confirm that the host can ping the robot. Put the MID-360 on the same `192.168.123.x` subnet, configure its addresses in Livox Viewer 2, and verify point-cloud output there before involving ROS. Then configure `sim2real/cyclone_config.xml` for the interface connected to the Go2:

```bash
export CYCLONEDDS_URI=file:///absolute/path/to/LoComposition/sim2real/cyclone_config.xml
export ROS_DOMAIN_ID=0
```

The selected interface must support multicast and have a multicast route. Validate the path with `ros2 multicast send`/`ros2 multicast receive`, restart the ROS daemon if discovery is stale, and confirm that `/lowstate` is actually receiving messages—not merely listed.

## Build the container and workspace

Allow local Docker GUI forwarding for Rviz:

```bash
xhost +local:docker
```

From the repository, run:

```bash
./sim2real/build-and-run.sh
```

Inside the container, bootstrap and build the ROS workspace:

```bash
/app/sim2real/bootstrap_ros2_ws.sh
source /app/ros2_ws/install/setup.bash
```

The bootstrap imports external packages listed in `sim2real/dependencies.repos`, including message and map dependencies that may not exist on a bare host ROS installation. Compose builds and runs `loukordos/locomposition-sim2real:latest` by default.

## Configure the MID-360 and robot transform

In the container, edit the imported Livox configuration so that host-IP fields refer to the machine running Docker and the LiDAR IP matches Livox Viewer:

```text
/app/ros2_ws/src/third_party/livox_ros_driver2/config/MID360_config.json
```

Set the measured LiDAR-to-base extrinsics in:

```text
/app/ros2_ws/src/locomposition_bringup/config/robot_real_go2.yaml
```

Bad extrinsics can look plausible while standing and fail as soon as the robot pitches or turns. Open `locomposition_bringup/rviz/go2_elevation_mapping.rviz`, move the robot through position and orientation changes, and verify that the physical ground remains flat and fixed in the map.

## Run elevation mapping

The stack expects the ROS 2 branch of [`elevation_mapping_cupy`](https://github.com/LouKordos/elevation_mapping_cupy/tree/ros2_humble) in a separate CUDA-enabled container. Keeping it separate avoids coupling the controller image to the mapping package's CUDA dependencies.

Use something similar to the commands below to set it up. (My fork has the Go2 setup already but is a bit outdated):

```bash
./src/elevation_mapping_cupy/docker/build.sh
source install/setup.bash
ros2 launch elevation_mapping_cupy elevation_mapping_go2.launch.py use_python_node:=false
```

Use host networking and the same CycloneDDS configuration. Ideally, use a deskewed and complete LiDAR scan because robot motion during an undeskewed scan can bend the terrain enough to corrupt policy observations. Configure the highest stable map update rate your hardware supports and use filters such as `min_filter` and `inpainting` to repair sparse cells before post-processing.

## Validate before enabling locomotion

With the robot still secured, check all of the following:

1. `/lowstate` updates continuously with plausible joint values and current timestamps.
2. Odometry, LiDAR, and base transforms form a connected TF tree without extrapolation errors.
3. The global elevation map updates while the robot moves.
4. The processed 13×11 map has finite values, expected orientation, and sensible heights relative to the base.
5. State and map age stay below the controller's freshness thresholds.

## Launch

After building and sourcing the workspace:

```bash
export ROS_DOMAIN_ID=0
ros2 launch locomposition_bringup bringup.launch.py
```

The launch uses Vicon odometry by default, but you can change this using the parameter `use_vicon`. Warnings containing `Failed to parse type hash for topic` can be filtered (reason is cyclonedds version mismatch on Go2) while diagnosing older Unitree message types, but do not filter other controller errors.