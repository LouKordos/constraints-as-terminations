from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    go2_odometry_launch_file = PathJoinSubstitution(
        [FindPackageShare("go2_odometry"), "launch", "go2_odometry_switch.launch.py"]
    )

    livox_launch_file = PathJoinSubstitution(
        [FindPackageShare("go2_livox_bringup"), "launch", "livox.launch.py"]
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([go2_odometry_launch_file]),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([livox_launch_file])
        ),
    ])