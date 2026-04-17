import os
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, ExecuteProcess, LogInfo, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare("cat_controller"),
        "config",
        "cat_control_node.yaml"
    ])

    pose_topic_arg = DeclareLaunchArgument(
        "pose_topic",
        description="Pose topic that the controller health check waits for. Passed from cat_bringup/launch/bringup.launch.py."
    )
    lowstate_topic_arg = DeclareLaunchArgument(
        "lowstate_topic",
        default_value="/lowstate",
        description="Low-level state topic that the controller health check waits for. Passed from cat_bringup/launch/bringup.launch.py."
    )
    livox_points_topic_arg = DeclareLaunchArgument(
        "livox_points_topic",
        default_value="/livox/lidar",
        description="Livox topic that the controller health check waits for. Passed from cat_bringup/launch/bringup.launch.py."
    )
    network_interface_arg = DeclareLaunchArgument(
        "network_interface",
        description="Network interface parameter override for cat_control_node. Passed from cat_bringup/launch/bringup.launch.py."
    )

    pose_topic = LaunchConfiguration("pose_topic")
    lowstate_topic = LaunchConfiguration("lowstate_topic")
    livox_points_topic = LaunchConfiguration("livox_points_topic")
    network_interface = LaunchConfiguration("network_interface")

    starting_log = LogInfo(
        msg=[
            "[HEALTH CHECK] Waiting for ",
            lowstate_topic,
            ", ",
            pose_topic,
            ", and ",
            livox_points_topic,
            " to publish data..."
        ]
    )

    health_check_cmd = ExecuteProcess(
        cmd=["sh", "-c", [
            "ros2 topic echo --once ", lowstate_topic, " > /dev/null && "
            "ros2 topic echo --once ", pose_topic, " > /dev/null && "
            "ros2 topic echo --once ", livox_points_topic, " > /dev/null"
        ]],
        output='log'
    )

    cat_control_node = Node(
        package="cat_controller",
        executable="cat_controller",
        name="cat_controller",
        output="screen",
        parameters=[config_path, {"network_interface": network_interface}]
    )

    start_controller_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=health_check_cmd,
            on_exit=[
                LogInfo(msg="[HEALTH CHECK PASSED] All sensors active. Starting CaT Control Node..."),
                cat_control_node
            ]
        )
    )

    return LaunchDescription([
        pose_topic_arg,
        lowstate_topic_arg,
        livox_points_topic_arg,
        network_interface_arg,
        starting_log,
        health_check_cmd,
        start_controller_event
    ])