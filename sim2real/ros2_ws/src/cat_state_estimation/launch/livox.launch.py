import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    livox_config_path = os.path.join(get_package_share_directory('livox_ros_driver2'), 'config', 'MID360_config.json')

    # Static transform from base to the livox_frame
    # Update these arguments with your measured physical offsets!
    # args: 'x y z yaw pitch roll parent_frame child_frame' (meters and radians)
    # TODO: Make params
    x_offset = 0.34
    y_offset = 0.0
    z_offset = 0.155
    roll_offset = 0.0
    pitch_offset = 0.784
    yaw_offset = 0.0
    
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
            {'publish_freq': 30.0}, # TODO: Make param
            {'frame_id': 'livox_frame'},
            {'user_config_path': livox_config_path}
        ]
    )

    return LaunchDescription([
        livox_driver_node
    ])