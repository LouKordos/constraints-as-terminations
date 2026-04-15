#pragma once

#include "cat_controller/stamped_robot_state.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

// Joint order in isaac lab is "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",
// "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
// Joint order reported by SDK state array is FR_hip_joint, FR_thigh_joint, FR_calf_joint, FL_hip_joint, FL_thigh_joint, FL_calf_joint,
// RR_hip_joint, RR_thigh_joint, RR_calf_joint, RL_hip_joint, RL_thigh_joint, RL_calf_joint
static constexpr int sdk_to_isaac_idx[12] = {
    /*0*/ 1,   // FR_hip → Isaac[1]
    /*1*/ 5,   // FR_thigh → Isaac[5]
    /*2*/ 9,   // FR_calf → Isaac[9]
    /*3*/ 0,   // FL_hip → Isaac[0]
    /*4*/ 4,   // FL_thigh → Isaac[4]
    /*5*/ 8,   // FL_calf → Isaac[8]
    /*6*/ 3,   // RR_hip → Isaac[3]
    /*7*/ 7,   // RR_thigh → Isaac[7]
    /*8*/ 11,  // RR_calf → Isaac[11]
    /*9*/ 2,   // RL_hip → Isaac[2]
    /*10*/ 6,  // RL_thigh → Isaac[6]
    /*11*/ 10  // RL_calf → Isaac[10]
};

// TODO: Move into helper class or replace with library
// Compute body-frame gravity vector given a body→world quaternion.
// quat_body_to_world_wxyz is a unit quaternion [w, x, y, z] rotating body to world.
// Returns [g_x, g_y, g_z] in body frame.
static inline std::array<float, 3> projected_gravity_body_frame(const std::array<float, 4> & quat_body_to_world_wxyz)
{
    // Extract components
    const float w = quat_body_to_world_wxyz[0];
    const float x = quat_body_to_world_wxyz[1];
    const float y = quat_body_to_world_wxyz[2];
    const float z = quat_body_to_world_wxyz[3];

    const float wi = w;
    const float xi = -x;
    const float yi = -y;
    const float zi = -z;

    // First Hamilton product: q_inv ⊗ g
    const float a0 = zi;
    const float a1 = -yi;
    const float a2 = xi;
    const float a3 = -wi;

    // Second Hamilton product: (q_inv ⊗ g) ⊗ q
    const float r1 = a0 * x + a1 * w + a2 * z - a3 * y;
    const float r2 = a0 * y - a1 * z + a2 * w + a3 * x;
    const float r3 = a0 * z + a1 * y - a2 * x + a3 * w;

    return {r1, r2, r3};
}

inline stamped_robot_state stamped_state_from_lowstate(
    const unitree_go::msg::LowState & lowstate, const long long & iteration_counter, const std::chrono::steady_clock::time_point & msg_publish_time)
{
    auto quat_wxyz_body_to_world = lowstate.imu_state.quaternion;
    stamped_robot_state stamped_state;

    stamped_state.foot_forces_raw = lowstate.foot_force;
    stamped_state.quat_body_to_world_wxyz = quat_wxyz_body_to_world;
    stamped_state.body_rpy_xyz = lowstate.imu_state.rpy;
    stamped_state.projected_gravity = projected_gravity_body_frame(quat_wxyz_body_to_world);
    stamped_state.body_angular_velocity = lowstate.imu_state.gyroscope;

    for (int i = 0; i < 12; i++) {
        int j = sdk_to_isaac_idx[i];
        stamped_state.joint_pos[j] = lowstate.motor_state[i].q;
        stamped_state.joint_vel[j] = lowstate.motor_state[i].dq;
        stamped_state.joint_torque[j] = lowstate.motor_state[i].tau_est;
    }
    stamped_state.timestamp = msg_publish_time;
    stamped_state.counter = iteration_counter;
}

// Init the message struct with appropriate default values. This modifies the message in place
inline void init_command_msg(unitree_go::msg::LowCmd & msg)
{
    msg.head[0] = 0xFE;
    msg.head[1] = 0xEF;
    msg.level_flag = 0xFF;
    msg.gpio = 0;

    for (int i = 0; i < 20; i++) {
        msg.motor_cmd[i].mode = 0x01;
        msg.motor_cmd[i].q = PosStopF;
        msg.motor_cmd[i].dq = VelStopF;
        msg.motor_cmd[i].kp = 0.0;
        msg.motor_cmd[i].kd = 0.0;
        msg.motor_cmd[i].tau = 0.0;
    }
}