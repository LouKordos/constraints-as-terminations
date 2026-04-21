# Constraints As Terminations (CaT)

[Website](https://constraints-as-terminations.github.io) | [Technical Paper](https://arxiv.org/abs/2403.18765) | [Videos](https://www.youtube.com/watch?v=crWoYTb8QvU)

![](assets/teaser.png)

## About this repository

This repository contains an Isaaclab implementation of the article **CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning** by Elliot Chane-Sane\*, Pierre-Alexandre Leziart\*, Thomas Flayols, Olivier Stasse, Philippe Souères, and Nicolas Mansard.

This implementation was built by Constant Roux and Maciej Stępień.

This paper has been accepted for the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024).

This code relies on the [CleanRL](https://github.com/vwxyzjn/cleanrl) library and [IsaacLab](https://isaac-sim.github.io/IsaacLab/v1.4.1/index.html) (version 1.4.1).

Implementation of the constraints manager and modification of the environment can be found in the [CaT directory](exts/cat_envs/cat_envs/tasks/utils/cat/). The modified PPO implementation can be found in the [CleanRL directory](exts/cat_envs/cat_envs/tasks/utils/cleanrl/).

`ConstraintsManager` follows the manager-based Isaac Lab approach, allowing easy integration just like other managers. For a full example, check out [cat_flat_env_cfg.py](exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_flat_env_cfg.py).

```python
@configclass
class ConstraintsCfg:
    # Safety Soft Constraints
    joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={"limit": 3.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    )
    # Safety Hard Constraints
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={"names": ["base_link", ".*_UPPER_LEG"]},
    )
```


## Running CaT (fork, follow this)

### Training and Isaac Lab Simulation

Training and Isaac lab-related tasks were tested without docker, by using `create-isaac-lab-env-uv.py`. Check the slurm config and the `justfile` for available commands, as well as the `scripts` directory.

## Sim2Real
For deploying on the real robot, the `sim2real` directory is relevant. It uses docker for ROS and you should build and run using `build-and-run.sh`. Check the options inside the script.

Initial setup steps:
1. Run `xhost +local:docker` on the host (outside docker) so that GUI applications such as RViz2 work through docker.
2. Set up your network interface and ensure you can ping the robot: [https://support.unitree.com/home/en/developer/Quick_start](https://support.unitree.com/home/en/developer/Quick_start). You can run the sim2real code on either a connected workstation or the Go2 itself, but the latter is recommended for latency reasons.

**Ensure the time between Go2 and your workstation are synced!** This is crucial for the ROS computations to work properly:
1. On the workstation connected to Go2: Add `allow 192.168.123.0/24` in `/etc/chrony/chrony.conf`
2. Allow NTP through the firewall (if enabled)
3. On the Go2: Add `server [YOUR_WORKSTATION_IP] iburst prefer minpoll 0 maxpoll` to `/etc/chrony/chrony.conf`
4. Confirm time is sufficiently accurate using `chronyc tracking`. Then, ensure you somehow have internet access on the Go2, e.g. using something like `sshuttle`.

To set up the MID360 LIDAR (used for elevation mapping), ensure the LIDAR is on the same subnet as the robot (`192.168.123.xxx`, can be done with [Livox Viewer 2](https://www.livoxtech.com/downloads)) and that you can ping it. Then, ensure data readouts work properly using [Livox Viewer 2](https://www.livoxtech.com/downloads).

### CycloneDDS and network configuration
There might be connectivity issues between workstation and robot depending on how your network interfaces are set up:
1. Edit the network interface in `sim2real/cyclone_config.xml` to the one that is connected to your Go2, so that CycloneDDS only sends traffic on this interface. 
2. Set `export CYCLONEDDS_URI=file://[PATH_TO_PROVIDED_CYCLONEDDS_CONFIG_IN_THIS_REPO]`
3. Ensure you have a multicast route on the network interface being used for robot communication (e.g. using `sudo ip route add 224.0.0.0/4 dev [YOUR_NETWORK_INTERFACE]`)
4. Ensure that `MULTICAST` shows up for the network interface you are using under `ip link show`
6. `ros2 daemon stop && ros2 daemon start`

To check if everything is set up properly: 
1. Ensure `ros2 multicast send` on workstation results in a received message when running `ros2 multicast receive` on the Go2. 
2. Check if `ros2 topic echo /lowstate` and other topics are printing out the robot's state correctly, or at least receiving messages.

### Run the policy
The `build-and-run.sh` script is responsible for building both the docker container and the actual code base, so you simply run it on the host after cloning the repository and then once more inside the container. The script will automatically enter a shell inside the container and it is recommended to use `tmux` so that OSC-52 can be used for copying via SSH. After `build-and-run.sh` has dropped you into the container shell:

1. Execute `build-and-run.sh --clean-build` inside the docker shell.
2. Adjust `/app/ros2_ws/src/third_party/livox_ros_driver2/config/MID360_config.json`: The host IP entries should be set to the machine that is running the docker container, the LIDAR IP should be the one you set in Livox Viewer (same subnet as robot).
3. Adjust extrinsics in `/app/ros2_ws/src/cat_bringup/config/robot_real_go2.yaml`, depending on how and where you mounted your LIDAR. This will create a static transform publisher to define where the lidar is relative to the base link.

Now you can launch the ROS nodes.
1. `/app/sim2real/bootstrap_ros2_ws.sh`
2. `export ROS_DOMAIN_ID=0` so that Go2 topics become visible
3. `source /app/ros2_ws/install/setup.bash`
4. Ensure you see `/lowstate` under `ros2 topic list`

To start the ROS setup, either run `/app/sim2real/build-and-run.sh`, or `ros2 launch cat_bringup bringup.launch.py | grep -v "Failed to parse type hash for topic"`. Odom uses vicon by default (adjustable in the launch arg file of cat_bringup), and requires [https://github.com/dasc-lab/ros2-vicon-bridge](https://github.com/dasc-lab/ros2-vicon-bridge). Lastly, open `ros2_ws/src/cat_bringup/rviz/go2_elevation_mapping.rviz` to inspect elevation map, pointcloud and tf tree.

### Elevation Mapping
This repo expects that you run [elevation_mapping_cupy](github.com/leggedrobotics/elevation_mapping_cupy/tree/ros2), specifically the `ros2` branch in a **separate container**. The reasoning behind this is a separation of dependencies, as the package relies on CUDA, but `cat_controller` does not. Simply follow their setup steps to get the container running with host network and cyclonedds, and ensure you can see the published elevation map topics from the controller container. As long as that is given, the map processing in this repo can be configured using `cat_elevation_map_processing_node.yaml` and convert the map into a format the policy was trained with.
### Important notes:
- `elevation_mapping_cupy` works best with a deskewed lidar topic, as dynamic motion during the scan can severely degrade the accuracy of the produced map. It does not perform any point aggregation before processing the lidar pointcloud, so full, deskewed, potentially less frequent scans are desirable.
- Ensure that your odometry and lidar data is highly reliable and accurate.
- You can visualize the resulting map using `cat_bringup/rviz/go2_elevation_mapping.rviz`, let the robot walk around and change orientation and check if the ground stays perfectly flat.
- Accurate and synced time is crucial for this stack, ideally you run everything on one machine. If not, confirm using `chronyc tracking` that you are below 1ms of drift.
- Configure the parameters inside of `elevation_mapping_cupy` to use the highest map update frequency possible on your hardware, same for any other `_fps` params.
- Use plugins such as `min_filter` `inpainting` to clean up cells before passing to the procesisng node. While NaN/inf are checked, using these plugins will produce much more reliable results if neighboring cells can be used to improve the overall map.

Example command for starting elevation_mapping_cupy: `./src/elevation_mapping_cupy/docker/build.sh && source install/setup.bash && ros2 launch elevation_mapping_cupy elevation_mapping_go2.launch.py use_python_node:=false`