import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Path to the Livox driver's config file
    livox_config_path = os.path.join(
        get_package_share_directory('livox_ros_driver2'), 'config', 'MID360_config.json'
    )

    return LaunchDescription([
        # Static transform from base to the livox_frame
        # Update these arguments with your measured physical offsets!
        # args: 'x y z yaw pitch roll parent_frame child_frame' (meters and radians)

        # Livox MID360 Driver Node
        Node(
            package='livox_ros_driver2',
            executable='livox_ros_driver2_node',
            name='livox_lidar_publisher',
            output='screen',
            parameters=[
                {'xfer_format': 0},
                {'frame_id': 'livox_frame'},
                {'user_config_path': livox_config_path}
            ]
        ),
    ])
