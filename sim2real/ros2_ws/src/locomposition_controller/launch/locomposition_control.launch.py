from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _handle_health_check_exit(event, context, controller_node):
    if event.returncode == 0:
        return [
            LogInfo(msg="[HEALTH CHECK PASSED] All sensors active. Starting LoComposition control node..."),
            controller_node,
        ]

    return [
        LogInfo(
            msg=f"[HEALTH CHECK FAILED] Topic wait helper exited with code {event.returncode}. Shutting down bringup."
        ),
        EmitEvent(event=Shutdown(reason="Controller health check failed")),
    ]


def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare("locomposition_controller"),
        "config",
        "locomposition_control_node.yaml"
    ])
    wait_for_topics_script = PathJoinSubstitution([
        FindPackageShare("locomposition_controller"),
        "launch",
        "wait_for_topics.py"
    ])

    pose_topic_arg = DeclareLaunchArgument(
        "pose_topic",
        description="Pose topic that the controller health check waits for. Passed from locomposition_bringup/launch/bringup.launch.py."
    )
    lowstate_topic_arg = DeclareLaunchArgument(
        "lowstate_topic",
        default_value="/lowstate",
        description="Low-level state topic that the controller health check waits for. Passed from locomposition_bringup/launch/bringup.launch.py."
    )
    livox_points_topic_arg = DeclareLaunchArgument(
        "livox_points_topic",
        default_value="/livox/lidar",
        description="Livox topic that the controller health check waits for. Passed from locomposition_bringup/launch/bringup.launch.py."
    )
    network_interface_arg = DeclareLaunchArgument(
        "network_interface",
        description="Network interface parameter override for locomposition_controller. Passed from locomposition_bringup/launch/bringup.launch.py."
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
            # pose_topic,
            # ", and ",
            # livox_points_topic,
            " to publish data..."
        ]
    )

    health_check_cmd = ExecuteProcess(
        cmd=[
            FindExecutable(name="python3"),
            wait_for_topics_script,
            "--timeout-sec",
            "30.0",
            lowstate_topic,
            # pose_topic,
            # livox_points_topic,
        ],
        output="both",
    )

    locomposition_control_node = Node(
        package="locomposition_controller",
        executable="locomposition_controller",
        name="locomposition_controller",
        output="both",
        # ros_arguments=["--log-level", "debug"],
        parameters=[config_path, {"network_interface": network_interface}]
    )

    start_controller_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=health_check_cmd,
            on_exit=lambda event, context: _handle_health_check_exit(event, context, locomposition_control_node)
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