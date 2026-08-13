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


def test_perception_message_package_is_canonical():
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
    assert "locomposition_perception_msgs" in canonical_text


def test_canonical_controller_entrypoints_are_locomposition_named():
    controller = ROS_SRC / "locomposition_controller"
    assert (controller / "launch" / "locomposition_control.launch.py").is_file()
    assert (controller / "config" / "locomposition_control_node.yaml").is_file()
    assert (controller / "include" / "locomposition_controller").is_dir()

    cmake = (controller / "CMakeLists.txt").read_text()
    assert "project(locomposition_controller)" in cmake
    assert "add_executable(locomposition_controller" in cmake


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
