"""Compatibility entry point for the former state-estimation package name."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    canonical_launch = PathJoinSubstitution(
        [
            FindPackageShare("locomposition_state_estimation"),
            "launch",
            "livox.launch.py",
        ]
    )
    return LaunchDescription(
        [IncludeLaunchDescription(PythonLaunchDescriptionSource(canonical_launch))]
    )
