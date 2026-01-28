import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node

def generate_launch_description():
    # Path to the Livox driver's config file
    livox_config_path = os.path.join(get_package_share_directory('livox_ros_driver2'), 'config', 'MID360_config.json')

    use_vicon_arg = DeclareLaunchArgument(
        "use_vicon",
        default_value="true",
        description="Decides which frame names to use for the static transform between lidar and base frame"
    )
    use_vicon = LaunchConfiguration("use_vicon")

    # Static transform from base to the livox_frame
    # Update these arguments with your measured physical offsets!
    # args: 'x y z yaw pitch roll parent_frame child_frame' (meters and radians)
    x_offset = 0.34
    y_offset = 0.0
    z_offset = 0.155
    roll_offset = 0.0
    pitch_offset = 0.784
    yaw_offset = 0.0
    vicon_base_frame = "vicon/Go2_Loukas/Go2"
    odometry_base_frame = "base"

    tf_vicon_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_livox_static_transform',
        arguments=[str(x_offset), str(y_offset), str(z_offset), str(yaw_offset), str(pitch_offset), str(roll_offset), vicon_base_frame, 'livox_frame'],
        output='screen',
        condition=IfCondition(use_vicon)
    )

    # This is needed so that the state publisher and the URDF find a transform for the base.
    # Vicon publishes vicon_base_frame as the base so we just need a transform with base in same position as vicon frame
    tf_vicon_base_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_vicon_static_transform",
        arguments=["0","0","0","0","0","0", vicon_base_frame, odometry_base_frame],
        output="screen",
        condition=IfCondition(use_vicon)
    )

    tf_odometry_node = Node( 
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_livox_static_transform',
        arguments=[str(x_offset), str(y_offset), str(z_offset), str(yaw_offset), str(pitch_offset), str(roll_offset), odometry_base_frame, 'livox_frame'],
        output='screen',
        condition=UnlessCondition(use_vicon)
    )
    
    livox_driver_node = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[
            {'xfer_format': 0},
            {'publish_freq': 30.0},
            {'frame_id': 'livox_frame'},
            {'user_config_path': livox_config_path}
        ]
    )

    return LaunchDescription([
        use_vicon_arg,
        tf_vicon_node,
        tf_vicon_base_node,
        tf_odometry_node,
        livox_driver_node
    ])
