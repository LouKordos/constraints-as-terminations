from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, FindExecutable
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node

def generate_launch_description():
    use_vicon_arg = DeclareLaunchArgument(
        "use_vicon",
        default_value="false",
        description="Set to true if you want to use vicon tracking instead of go2_odometry. This simply adjusts the static transform being published for the lidar and starts only the robot state publisher instead of full go2_odometry."
    )
    use_vicon = LaunchConfiguration("use_vicon")

    livox_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_state_estimation"), "launch", "livox.launch.py"]
    )
    livox_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([livox_launch_file]),
        launch_arguments={"use_vicon": use_vicon}.items()
    )

    odom_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_state_estimation"), "launch", "odom.launch.py"]
    )
    odom_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([odom_launch_file]),
        launch_arguments={"use_vicon": use_vicon}.items()
    )

    controller_launch_file = PathJoinSubstitution(
        [FindPackageShare("cat_controller"), "launch", "cat_control.launch.py"]
    )
    controller_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([controller_launch_file]),
        launch_arguments={"use_vicon": use_vicon}.items()
    )

    return LaunchDescription([
        use_vicon_arg,
        livox_include,
        odom_include,
        controller_include
    ])