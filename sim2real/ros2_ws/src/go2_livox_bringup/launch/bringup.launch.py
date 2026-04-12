from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, FindExecutable

from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from go2_description import GO2_DESCRIPTION_URDF_PATH

def generate_launch_description():
    go2_odometry_launch_file = PathJoinSubstitution(
        [FindPackageShare("go2_odometry"), "launch", "go2_odometry_switch.launch.py"]
    )

    livox_launch_file = PathJoinSubstitution(
        [FindPackageShare("go2_livox_bringup"), "launch", "livox.launch.py"]
    )

    with open(GO2_DESCRIPTION_URDF_PATH, "r") as info:
        robot_desc = info.read()

    use_vicon_arg = DeclareLaunchArgument(
        "use_vicon",
        default_value="false",
        description="Set to true if you want to use vicon tracking instead of go2_odometry. This simply adjusts the static transform being published for the lidar and starts only the robot state publisher instead of full go2_odometry."
    )
    use_vicon = LaunchConfiguration("use_vicon")

    livox_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([livox_launch_file]),
        launch_arguments={"use_vicon": use_vicon}.items()
    )
    odometry_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([go2_odometry_launch_file]),
        condition=UnlessCondition(use_vicon)        
    )

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

    # Temporary because policy inference is not connected to ros so we need to convert to zmq message
    script_path = "/app/odom_alternative_ws/src/temp_joystick_command_zmq.py"
    joystick_zmq_translation_node = ExecuteProcess(cmd=[FindExecutable(name="python3"), script_path], output="screen", log_cmd=True)

    return LaunchDescription([
        use_vicon_arg,
        livox_include,
        odometry_include,
        state_publisher_node,
        state_converter_node,
        joystick_zmq_translation_node
    ])
