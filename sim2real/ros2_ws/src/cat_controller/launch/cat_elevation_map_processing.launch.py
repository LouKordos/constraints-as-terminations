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

def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare("cat_controller"),
        "config",
        "cat_elevation_map_processing_node.yaml"
    ])

    elevation_map_processing_node = Node(
        package="cat_controller",
        executable="elevation_map_processing_node",
        name="elevation_map_processing_node",
        output="both",
        ros_arguments=["--log-level", "debug"],
        parameters=[config_path]
    )

    return LaunchDescription([
        elevation_map_processing_node
    ])