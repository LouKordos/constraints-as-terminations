from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROS_SRC = ROOT / "sim2real" / "ros2_ws" / "src"


def manifest_name(package_directory: Path) -> str | None:
    root = ET.parse(package_directory / "package.xml").getroot()
    return root.findtext("name")


def tree_text(directory: Path) -> str:
    return "\n".join(
        path.read_text(errors="ignore")
        for path in directory.rglob("*")
        if path.is_file()
    )


def test_canonical_ros_package_directories_match_manifest_names():
    package_names = (
        "locomposition_controller",
        "locomposition_bringup",
        "locomposition_state_estimation",
        "locomposition_perception_msgs",
    )

    for package_name in package_names:
        package_directory = ROS_SRC / package_name
        assert package_directory.is_dir()
        assert manifest_name(package_directory) == package_name


def test_perception_message_type_moved_without_a_wire_compatibility_claim():
    assert not (ROS_SRC / "cat_perception_msgs").exists()
    message = (
        ROS_SRC
        / "locomposition_perception_msgs"
        / "msg"
        / "ProcessedElevationMap.msg"
    )
    assert message.is_file()

    canonical_text = "\n".join(
        tree_text(ROS_SRC / package_name)
        for package_name in (
            "locomposition_controller",
            "locomposition_bringup",
            "locomposition_state_estimation",
            "locomposition_perception_msgs",
        )
    )
    assert "cat_perception_msgs" not in canonical_text
    assert "locomposition_perception_msgs" in canonical_text


def test_canonical_controller_entrypoints_are_locomposition_named():
    controller = ROS_SRC / "locomposition_controller"
    assert (controller / "launch" / "locomposition_control.launch.py").is_file()
    assert (controller / "config" / "locomposition_control_node.yaml").is_file()
    assert (controller / "include" / "locomposition_controller").is_dir()

    cmake = (controller / "CMakeLists.txt").read_text()
    assert "project(locomposition_controller)" in cmake
    assert "add_executable(locomposition_controller" in cmake


def test_legacy_non_message_packages_are_launch_only_forwarders():
    expected_launch_files = {
        "cat_controller": {
            "cat_control.launch.py",
            "cat_elevation_map_processing.launch.py",
            "cat_elevation_map_comparison.launch.py",
        },
        "cat_bringup": {"bringup.launch.py"},
        "cat_state_estimation": {"livox.launch.py", "odom.launch.py"},
    }

    for package_name, launch_files in expected_launch_files.items():
        package_directory = ROS_SRC / package_name
        assert manifest_name(package_directory) == package_name
        assert not (package_directory / "src").exists()
        assert not (package_directory / "config").exists()
        assert {
            path.name for path in (package_directory / "launch").glob("*.launch.py")
        } == launch_files
        for launch_file in launch_files:
            wrapper = (package_directory / "launch" / launch_file).read_text()
            assert "IncludeLaunchDescription" in wrapper
            assert "locomposition_" in wrapper


def test_sim2real_entrypoints_default_to_locomposition_names():
    build_script = (ROOT / "sim2real" / "build-and-run.sh").read_text()
    bootstrap = (ROOT / "sim2real" / "bootstrap_ros2_ws.sh").read_text()
    compose = (ROOT / "compose.yml").read_text()

    assert "ros2 launch locomposition_bringup bringup.launch.py" in build_script
    assert "locomposition_controller/include/locomposition_controller" in bootstrap
    assert "LOCOMPOSITION_SIM2REAL_IMAGE" in compose


def test_generated_controller_crc_sources_stay_ignored_after_the_rename():
    generated_paths = (
        "sim2real/ros2_ws/src/locomposition_controller/"
        "include/locomposition_controller/motor_crc.h",
        "sim2real/ros2_ws/src/locomposition_controller/src/motor_crc.cpp",
    )

    for generated_path in generated_paths:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", generated_path],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
