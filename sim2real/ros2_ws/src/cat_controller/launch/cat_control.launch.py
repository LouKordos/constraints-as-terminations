import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, ExecuteProcess, LogInfo, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare("cat_controller"),
        "config",
        "cat_control_node.yaml"
    ])

    use_vicon_arg = DeclareLaunchArgument(
        "use_vicon",
        default_value="false",
        description="Determines which state topic to wait for."
    )
    use_vicon = LaunchConfiguration("use_vicon")

    starting_log = LogInfo(
        msg="[HEALTH CHECK] Waiting for Odometry/Vicon and Livox topics to publish data..."
    )
    odom_topic = PythonExpression([
        "'vicon/Go2_Loukas/Go2_Loukas/pose' if '", use_vicon, "' == 'true' else '/odometry/filtered'"
    ])

    health_check_cmd = ExecuteProcess(
        cmd=["sh", "-c", [
            "ros2 topic echo --once /lowstate > /dev/null && "
            "ros2 topic echo --once ", odom_topic, " > /dev/null && ",
            "ros2 topic echo --once /livox/lidar > /dev/null"
        ]],
        output='log'
    )

    cat_control_node = Node(
        package="cat_controller",
        executable="cat_control_node",
        name="cat_control_node",
        output="screen",
        parameters=[config_path]
    )

    start_controller_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=health_check_cmd,
            on_exit=[
                LogInfo(msg="[HEALTH CHECK PASSED] All sensors active. Starting CaT Control Node..."),
                cat_control_node
            ]
        )
    )

    return LaunchDescription([
        use_vicon_arg,
        starting_log,
        health_check_cmd,
        start_controller_event
    ])