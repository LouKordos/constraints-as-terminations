#include <array>
#include <chrono>

// Wrapper for robot observation state with timestamp to detect stale/outdated data.
// Based on https://support.unitree.com/home/en/developer/Basic_services
struct stamped_robot_state{
    std::array<int16_t, 4> foot_forces_raw; // Raw values ranging from 0 to 4096
    std::array<float, 4> quat_body_to_world_wxyz; // WXYZ Quaternion from body to world frame
    std::array<float, 3> body_rpy_xyz; // Euler angles (roll pitch yaw) in xyz order (from what I can tell from the unitree docs)
    std::array<float, 3> projected_gravity; // Projected gravity vector (0,0,-1) expressed in body frame
    std::array<float, 3> body_angular_velocity; // XYZ Body angular velocity (rad/s)
    std::array<float, 12> joint_pos; // Joint positions (q) for each joint. Order as defined in IsaacLab env ObservationsCfg
    std::array<float, 12> joint_vel; // Joint positions (dq) for each joint. Order as defined in IsaacLab env ObservationsCfg
    std::chrono::steady_clock::time_point timestamp;
};