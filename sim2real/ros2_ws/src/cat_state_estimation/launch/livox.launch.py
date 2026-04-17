import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    livox_frame_arg = DeclareLaunchArgument(
        "livox_frame",
        default_value="livox_frame",
        description="Livox frame name passed from cat_bringup/launch/bringup.launch.py."
    )
    livox_publish_freq_arg = DeclareLaunchArgument(
        "livox_publish_freq",
        default_value="30.0",
        description="Livox publish frequency in Hz passed from cat_bringup/launch/bringup.launch.py."
    )

    livox_frame = LaunchConfiguration("livox_frame")
    livox_publish_freq = LaunchConfiguration("livox_publish_freq")

    livox_config_path = PathJoinSubstitution([
        FindPackageShare("livox_ros_driver2"), "config", "MID360_config.json"
    ])
    livox_driver_node = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[
            {'xfer_format': 0},
            {'publish_freq': livox_publish_freq},
            {'frame_id': livox_frame},
            {'user_config_path': livox_config_path}
        ]
    )

    return LaunchDescription([
        livox_frame_arg,
        livox_publish_freq_arg,
        livox_driver_node
    ])