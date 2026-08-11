"""Compatibility entry point for the former map-comparison launch name."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    canonical_launch = PathJoinSubstitution(
        [
            FindPackageShare("locomposition_controller"),
            "launch",
            "locomposition_elevation_map_comparison.launch.py",
        ]
    )
    return LaunchDescription(
        [IncludeLaunchDescription(PythonLaunchDescriptionSource(canonical_launch))]
    )
