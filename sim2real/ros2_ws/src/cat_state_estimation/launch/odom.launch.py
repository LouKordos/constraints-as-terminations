import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from unitree_description import GO2_DESCRIPTION_URDF_PATH

def generate_launch_description():
    use_vicon_arg = DeclareLaunchArgument(
        "use_vicon",
        default_value="false",
        description="Decides which frame names to use for the static transform between lidar and base frame"
    )
    base_frame_arg = DeclareLaunchArgument(
        "base_frame",
        default_value="base",
        description="Base frame name passed from cat_bringup/launch/bringup.launch.py."
    )
    livox_frame_arg = DeclareLaunchArgument(
        "livox_frame",
        default_value="livox_frame",
        description="Livox frame name passed from cat_bringup/launch/bringup.launch.py."
    )
    vicon_base_frame_arg = DeclareLaunchArgument(
        "vicon_base_frame",
        default_value="vicon/Go2_Loukas/Go2_Loukas",
        description="Vicon base frame name passed from cat_bringup/launch/bringup.launch.py."
    )
    base_to_livox_x_arg = DeclareLaunchArgument(
        "base_to_livox_x",
        default_value="0.34",
        description="Base-to-Livox X offset passed from cat_bringup/launch/bringup.launch.py."
    )
    base_to_livox_y_arg = DeclareLaunchArgument(
        "base_to_livox_y",
        default_value="0.0",
        description="Base-to-Livox Y offset passed from cat_bringup/launch/bringup.launch.py."
    )
    base_to_livox_z_arg = DeclareLaunchArgument(
        "base_to_livox_z",
        default_value="0.155",
        description="Base-to-Livox Z offset passed from cat_bringup/launch/bringup.launch.py."
    )
    base_to_livox_yaw_arg = DeclareLaunchArgument(
        "base_to_livox_yaw",
        default_value="0.0",
        description="Base-to-Livox yaw passed from cat_bringup/launch/bringup.launch.py."
    )
    base_to_livox_pitch_arg = DeclareLaunchArgument(
        "base_to_livox_pitch",
        default_value="0.784",
        description="Base-to-Livox pitch passed from cat_bringup/launch/bringup.launch.py."
    )
    base_to_livox_roll_arg = DeclareLaunchArgument(
        "base_to_livox_roll",
        default_value="0.0",
        description="Base-to-Livox roll passed from cat_bringup/launch/bringup.launch.py."
    )

    use_vicon = LaunchConfiguration("use_vicon")
    vicon_base_frame = LaunchConfiguration("vicon_base_frame")
    odometry_base_frame = LaunchConfiguration("base_frame")
    livox_frame = LaunchConfiguration("livox_frame")
    x_offset = LaunchConfiguration("base_to_livox_x")
    y_offset = LaunchConfiguration("base_to_livox_y")
    z_offset = LaunchConfiguration("base_to_livox_z")
    yaw_offset = LaunchConfiguration("base_to_livox_yaw")
    pitch_offset = LaunchConfiguration("base_to_livox_pitch")
    roll_offset = LaunchConfiguration("base_to_livox_roll")

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
        arguments=[x_offset, y_offset, z_offset, yaw_offset, pitch_offset, roll_offset, vicon_base_frame, livox_frame],
        output='both',
        # ros_arguments=["--log-level", "debug"],
        condition=IfCondition(use_vicon)
    )

    # This is needed so that the state publisher and the URDF find a transform for the base.
    # Vicon publishes vicon_base_frame as the base so we just need a transform with base in same position as vicon frame
    tf_vicon_base_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_vicon_static_transform",
        arguments=["0","0","0","0","0","0", vicon_base_frame, odometry_base_frame],
        output="both",
        # ros_arguments=["--log-level", "debug"],
        condition=IfCondition(use_vicon)
    )

    tf_base_to_lidar_node = Node( 
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_livox_static_transform',
        arguments=[x_offset, y_offset, z_offset, yaw_offset, pitch_offset, roll_offset, odometry_base_frame, livox_frame],
        output='both',
        # ros_arguments=["--log-level", "debug"],
        condition=UnlessCondition(use_vicon)
    )

    # This isn't really best practice, but directly copied from go2_odometry to get the robot model showing up in RViz
    with open(GO2_DESCRIPTION_URDF_PATH, "r") as info:
        robot_desc = info.read()
    state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": robot_desc}],
        # ros_arguments=["--log-level", "debug"],
        arguments=[GO2_DESCRIPTION_URDF_PATH],
        condition=IfCondition(use_vicon)
    )

    # Converts /lowstate into a standard ROS message format, check go2_odometry package from INRIA for details
    state_converter_node = Node(
        package="go2_odometry",
        executable="state_converter_node",
        name="state_converter_node",
        # ros_arguments=["--log-level", "debug"],
        parameters=[],
        output="both",
        condition=IfCondition(use_vicon)
    )

    return LaunchDescription([
        use_vicon_arg,
        base_frame_arg,
        livox_frame_arg,
        vicon_base_frame_arg,
        base_to_livox_x_arg,
        base_to_livox_y_arg,
        base_to_livox_z_arg,
        base_to_livox_yaw_arg,
        base_to_livox_pitch_arg,
        base_to_livox_roll_arg,
        tf_vicon_node,
        tf_vicon_base_node,
        tf_base_to_lidar_node,
        go2_odometry_include,
        state_publisher_node,
        state_converter_node
    ])