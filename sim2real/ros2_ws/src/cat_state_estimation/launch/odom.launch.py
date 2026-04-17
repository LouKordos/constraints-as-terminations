import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from unitree_description import GO2_DESCRIPTION_URDF_PATH

def generate_launch_description():
    # TODO: Make parameters
    vicon_base_frame = "vicon/Go2_Loukas/Go2_Loukas"
    odometry_base_frame = "base"

    use_vicon_arg = DeclareLaunchArgument(
        "use_vicon",
        default_value="true",
        description="Decides which frame names to use for the static transform between lidar and base frame"
    )
    use_vicon = LaunchConfiguration("use_vicon")

    go2_odometry_launch_file = PathJoinSubstitution(
        [FindPackageShare("go2_odometry"), "launch", "go2_odometry_switch.launch.py"]
    )
    go2_odometry_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([go2_odometry_launch_file]),
        condition=UnlessCondition(use_vicon)        
    )

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

    # This isn't really best practice, but directly copied from go2_odometry to get the robot model showing up in RViz
    with open(GO2_DESCRIPTION_URDF_PATH, "r") as info:
        robot_desc = info.read()
    state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc}],
        arguments=[GO2_DESCRIPTION_URDF_PATH],
        condition=IfCondition(use_vicon)
    )

    state_converter_node = Node(
        package="go2_odometry",
        executable="state_converter_node",
        name="state_converter_node",
        parameters=[],
        output="screen",
        condition=IfCondition(use_vicon)
    )

    return LaunchDescription([
        use_vicon_arg,
        tf_vicon_node,
        tf_vicon_base_node,
        tf_odometry_node,
        go2_odometry_include
    ])