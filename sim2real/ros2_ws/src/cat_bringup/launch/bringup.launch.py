from datetime import datetime, timezone
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetROSLogDir
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *args, **kwargs):
    use_vicon = LaunchConfiguration("use_vicon")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = Path("/app/logs/ros") / f"utc_{timestamp}"
    bag_dir = run_dir / "bag"

    run_dir.mkdir(parents=True, exist_ok=False)
    bag_dir.mkdir(parents=True, exist_ok=False)

    livox_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_state_estimation"), "launch", "livox.launch.py"]
    )
    livox_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([livox_launch_file]),
    )

    odom_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_state_estimation"), "launch", "odom.launch.py"]
    )
    odom_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([odom_launch_file]),
        launch_arguments={"use_vicon": use_vicon}.items(),
    )

    controller_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_controller"), "launch", "cat_control.launch.py"]
    )
    controller_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([controller_launch_file]),
        launch_arguments={"use_vicon": use_vicon}.items(),
    )

    rosbag_record = ExecuteProcess(
        cmd=[
            FindExecutable(name="ros2"),
            "bag",
            "record",
            "--output",
            str(bag_dir),
            "/rosout",
            "/tf",
            "/tf_static",
            "/joint_states",
            "/lowstate",
            "/lowcmd",
            "/robot_description",
            "/initialpose",
            "/imu_lowstate",
            "/imu",
            "/odometry/filtered",
            "/livox/lidar",
            "/livox/imu",
            "/statistics",
            "/elevation_map_points",
            "/elevation_mapping_node/elevation_map_filter",
            "--regex",
            "^/[oO]dom.*",
        ],
        output="screen",
        sigterm_timeout="5",
        sigkill_timeout="5",
    )

    return [
        LogInfo(msg=f"ROS run directory: {run_dir}"),
        LogInfo(msg=f"ROS bag directory: {bag_dir}"),
        # Intentionally not wrapped in a GroupAction, since we want this ROS_LOG_DIR override to apply to all later-started processes
        # in this bringup launch, including the rosbag recorder and the nodes started by the included launch files.
        # This must come before them, because it only affects processes started after this action executes.
        SetROSLogDir(log_dir=str(run_dir)),
        rosbag_record,
        livox_include,
        odom_include,
        controller_include,
    ]

def generate_launch_description():
    use_vicon_arg = DeclareLaunchArgument(
        "use_vicon",
        default_value="false",
        description=(
            "Set to true if you want to use vicon tracking instead of go2_odometry. "
            "This simply adjusts the static transform being published for the lidar "
            "and starts only the robot state publisher instead of full go2_odometry."
        ),
    )

    return LaunchDescription(
        [
            use_vicon_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )