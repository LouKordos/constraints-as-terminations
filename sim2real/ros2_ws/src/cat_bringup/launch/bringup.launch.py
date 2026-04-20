from datetime import datetime, timezone
from pathlib import Path
import yaml

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import SetROSLogDir
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare


def _load_robot_config():
    config_path = Path(get_package_share_directory("cat_bringup")) / "config" / "robot_real_go2.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _launch_setup(context, *args, **kwargs):
    use_vicon = LaunchConfiguration("use_vicon")
    livox_frame = LaunchConfiguration("livox_frame")
    livox_publish_freq = LaunchConfiguration("livox_publish_freq")
    base_frame = LaunchConfiguration("base_frame")
    vicon_base_frame = LaunchConfiguration("vicon_base_frame")
    odom_pose_topic = LaunchConfiguration("odom_pose_topic")
    vicon_pose_topic = LaunchConfiguration("vicon_pose_topic")
    lowstate_topic = LaunchConfiguration("lowstate_topic")
    livox_points_topic = LaunchConfiguration("livox_points_topic")
    network_interface = LaunchConfiguration("network_interface")
    base_to_livox_x = LaunchConfiguration("base_to_livox_x")
    base_to_livox_y = LaunchConfiguration("base_to_livox_y")
    base_to_livox_z = LaunchConfiguration("base_to_livox_z")
    base_to_livox_yaw = LaunchConfiguration("base_to_livox_yaw")
    base_to_livox_pitch = LaunchConfiguration("base_to_livox_pitch")
    base_to_livox_roll = LaunchConfiguration("base_to_livox_roll")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = Path("/app/logs/ros") / f"utc_{timestamp}"
    bag_dir = run_dir / "bag"

    run_dir.mkdir(parents=True, exist_ok=False)

    livox_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_state_estimation"), "launch", "livox.launch.py"]
    )
    livox_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([livox_launch_file]),
        launch_arguments={
            "livox_frame": livox_frame,
            "livox_publish_freq": livox_publish_freq,
        }.items(),
    )

    odom_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_state_estimation"), "launch", "odom.launch.py"]
    )
    odom_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([odom_launch_file]),
        launch_arguments={
            "use_vicon": use_vicon,
            "base_frame": base_frame,
            "livox_frame": livox_frame,
            "vicon_base_frame": vicon_base_frame,
            "base_to_livox_x": base_to_livox_x,
            "base_to_livox_y": base_to_livox_y,
            "base_to_livox_z": base_to_livox_z,
            "base_to_livox_yaw": base_to_livox_yaw,
            "base_to_livox_pitch": base_to_livox_pitch,
            "base_to_livox_roll": base_to_livox_roll,
        }.items(),
    )

    controller_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_controller"), "launch", "cat_control.launch.py"]
    )
    # Resolve the single pose topic that the controller health check should wait for, then pass it into
    # cat_controller/launch/cat_control.launch.py so that launch file does not need to know how Vicon vs odometry is selected.
    controller_pose_topic = PythonExpression([
        "'", vicon_pose_topic, "' if '", use_vicon, "' == 'true' else '", odom_pose_topic, "'"
    ])
    controller_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([controller_launch_file]),
        launch_arguments={
            "pose_topic": controller_pose_topic,
            "lowstate_topic": lowstate_topic,
            "livox_points_topic": livox_points_topic,
            "network_interface": network_interface,
        }.items(),
    )

    elevation_map_processing_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_controller"), "launch", "cat_elevation_map_processing.launch.py"]
    )
    elevation_map_processing_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([elevation_map_processing_launch_file]),
    )

    elevation_map_comparison_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_controller"), "launch", "cat_elevation_map_comparison.launch.py"]
    )
    elevation_map_comparison_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([elevation_map_comparison_launch_file]),
    )

    rosbag_record = ExecuteProcess(
        cmd=[
            FindExecutable(name="ros2"),
            "bag",
            "record",
            "--output",
            str(bag_dir),
            "--topics",
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
            "/processed_elevation_map",
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
        SetROSLogDir(new_log_dir=str(run_dir)),
        rosbag_record,
        livox_include,
        odom_include,
        elevation_map_processing_include,
        elevation_map_comparison_include,
        controller_include,
    ]

def generate_launch_description():
    robot_config = _load_robot_config()

    use_vicon_arg = DeclareLaunchArgument(
        "use_vicon",
        default_value=str(robot_config["use_vicon"]).lower(),
        description="Set whether Vicon or go2_odometry is used. Passed to cat_state_estimation/odom.launch.py and used here to resolve the controller pose topic.",
    )
    livox_publish_freq_arg = DeclareLaunchArgument(
        "livox_publish_freq",
        default_value=str(robot_config["livox_publish_freq"]),
        description="Livox publish frequency in Hz. Passed to cat_state_estimation/livox.launch.py.",
    )
    network_interface_arg = DeclareLaunchArgument(
        "network_interface",
        default_value=robot_config["network_interface"],
        description="Network interface used by the controller node. Passed to cat_controller/launch/cat_control.launch.py.",
    )

    base_frame_arg = DeclareLaunchArgument(
        "base_frame",
        default_value=robot_config["frames"]["base"],
        description="Robot base frame name. Passed to cat_state_estimation/odom.launch.py.",
    )
    livox_frame_arg = DeclareLaunchArgument(
        "livox_frame",
        default_value=robot_config["frames"]["livox"],
        description="Livox frame name. Passed to cat_state_estimation/livox.launch.py and cat_state_estimation/odom.launch.py.",
    )
    vicon_base_frame_arg = DeclareLaunchArgument(
        "vicon_base_frame",
        default_value=robot_config["frames"]["vicon_base"],
        description="Vicon base frame name. Passed to cat_state_estimation/odom.launch.py.",
    )

    odom_pose_topic_arg = DeclareLaunchArgument(
        "odom_pose_topic",
        default_value=robot_config["topics"]["odom_pose"],
        description="Pose topic to use when use_vicon is false. Used here to resolve the controller pose topic.",
    )
    vicon_pose_topic_arg = DeclareLaunchArgument(
        "vicon_pose_topic",
        default_value=robot_config["topics"]["vicon_pose"],
        description="Pose topic to use when use_vicon is true. Used here to resolve the controller pose topic.",
    )
    lowstate_topic_arg = DeclareLaunchArgument(
        "lowstate_topic",
        default_value=robot_config["topics"]["lowstate"],
        description="Low-level robot state topic. Passed to cat_controller/launch/cat_control.launch.py.",
    )
    livox_points_topic_arg = DeclareLaunchArgument(
        "livox_points_topic",
        default_value=robot_config["topics"]["livox_points"],
        description="Livox point cloud topic. Passed to cat_controller/launch/cat_control.launch.py.",
    )

    base_to_livox_x_arg = DeclareLaunchArgument(
        "base_to_livox_x",
        default_value=str(robot_config["transforms"]["base_to_livox"]["x"]),
        description="Base-to-Livox static transform X offset. Passed to cat_state_estimation/odom.launch.py.",
    )
    base_to_livox_y_arg = DeclareLaunchArgument(
        "base_to_livox_y",
        default_value=str(robot_config["transforms"]["base_to_livox"]["y"]),
        description="Base-to-Livox static transform Y offset. Passed to cat_state_estimation/odom.launch.py.",
    )
    base_to_livox_z_arg = DeclareLaunchArgument(
        "base_to_livox_z",
        default_value=str(robot_config["transforms"]["base_to_livox"]["z"]),
        description="Base-to-Livox static transform Z offset. Passed to cat_state_estimation/odom.launch.py.",
    )
    base_to_livox_yaw_arg = DeclareLaunchArgument(
        "base_to_livox_yaw",
        default_value=str(robot_config["transforms"]["base_to_livox"]["yaw"]),
        description="Base-to-Livox static transform yaw. Passed to cat_state_estimation/odom.launch.py.",
    )
    base_to_livox_pitch_arg = DeclareLaunchArgument(
        "base_to_livox_pitch",
        default_value=str(robot_config["transforms"]["base_to_livox"]["pitch"]),
        description="Base-to-Livox static transform pitch. Passed to cat_state_estimation/odom.launch.py.",
    )
    base_to_livox_roll_arg = DeclareLaunchArgument(
        "base_to_livox_roll",
        default_value=str(robot_config["transforms"]["base_to_livox"]["roll"]),
        description="Base-to-Livox static transform roll. Passed to cat_state_estimation/odom.launch.py.",
    )

    return LaunchDescription(
        [
            use_vicon_arg,
            livox_publish_freq_arg,
            network_interface_arg,
            base_frame_arg,
            livox_frame_arg,
            vicon_base_frame_arg,
            odom_pose_topic_arg,
            vicon_pose_topic_arg,
            lowstate_topic_arg,
            livox_points_topic_arg,
            base_to_livox_x_arg,
            base_to_livox_y_arg,
            base_to_livox_z_arg,
            base_to_livox_yaw_arg,
            base_to_livox_pitch_arg,
            base_to_livox_roll_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )