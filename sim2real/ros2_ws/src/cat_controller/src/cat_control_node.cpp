#include <fmt/core.h>
#include <fmt/ranges.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <string>

#include "cat_controller/history_buffer.hpp"
#include "cat_controller/low_level_mode_enabler.hpp"
#include "cat_controller/motor_crc.h"  // Copied from go2 repo because its needed for sending valid motor commands and they do not install these header files automatically
#include "cat_controller/shutdown_coordinator.hpp"
#include "cat_controller/stamped_robot_state.hpp"
#include "cat_controller/time_utils.hpp"
#include "cat_controller/timed_atomic.hpp"
#include "cat_controller/unitree_msg_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

using namespace std::chrono_literals;

class CaTControlNode : public rclcpp::Node
{
public:
    // TODO: Move release_motion_mode.cpp and binary into ros package and find it relative to node executable
    explicit CaTControlNode(const std::string & network_interface)
        : Node("cat_control_node"),
          network_interface_(network_interface),  // TODO: Make this a ros param
          low_level_mode_enabler_("/app/sim2real/build/src/release_motion_mode", network_interface_, 45.0),
          // This handles threadsafe exit, avoids race conditions when exiting, and takes a lambda for cleanup that is called only once in the end.
          // Any node can request a shutdown using shutdown_coordinator.shutdown();
          shutdown_coordinator_(this->get_logger(), this->get_node_base_interface()->get_context(), [this]() {
              // Very important to put any cleanup for the node here!
              if (command_timer_) { command_timer_->cancel(); }
              if (policy_inference_timer_) { policy_inference_timer_->cancel(); }
              low_level_mode_enabler_.stop();
          })
    {
        static_assert(std::atomic<bool>::is_always_lock_free, "atomic bool is not lock free.");
        init_command_msg(command_msg_);
        load_pytorch_checkpoint();

        RCLCPP_DEBUG(this->get_logger(), "Starting robot state subscriber.");
        robot_state_sub_ = this->create_subscription<unitree_go::msg::LowState>("/lowstate", rclcpp::SensorDataQoS(),
            std::bind(&CaTControlNode::robot_state_callback, this, std::placeholders::_1, std::placeholders::_2));
        RCLCPP_DEBUG(this->get_logger(), "Started robot state subscriber.");

        RCLCPP_DEBUG(this->get_logger(), "Starting robot command publisher.");
        command_publisher = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", rclcpp::SensorDataQoS());
        RCLCPP_DEBUG(this->get_logger(), "Started robot command publisher.");

        std::string error_message;
        RCLCPP_DEBUG(this->get_logger(), "Starting low level control mode enabler process.");
        if (!low_level_mode_enabler_.start(error_message)) {
            shutdown_coordinator_.shutdown(error_message);
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Started motion switcher helper using interface '%s'.", network_interface_.c_str());

        RCLCPP_DEBUG(this->get_logger(), "Starting robot command publish timer.");
        command_timer_ = this->create_wall_timer(2ms, std::bind(&CaTControlNode::publish_commands, this));
        RCLCPP_DEBUG(this->get_logger(), "Started robot command publish timer.");

        RCLCPP_DEBUG(this->get_logger(), "Starting policy inference / control loop timer.");
        policy_inference_timer_ = this->create_wall_timer(20ms, std::bind(&CaTControlNode::policy_inference_callback, this));
        RCLCPP_DEBUG(this->get_logger(), "Started policy inference / control loop timer.");
        // Important TODO: Add linear interpolation from start pos to standing pos with Kp = 30 and Kd = 1 same way as run_policy.cpp
        // Important TODO: Set default global position targets used by publisher to be current position to avoid sudden movements while inference code
        // is not producing anything
    }

    // TODO: Move to ROS2 params
    std::chrono::microseconds atomic_op_timeout_threshold{500};
    const bool walk_a_bit = true;
    bool use_hardcoded_elevation = false;
    double hardcoded_elevation = -0.3f;
    const static short num_joints = 12;
    int observation_dim_no_history = 188;
    int observation_dim_history = 236;
    int history_length = 3;
    const float action_scale = 0.8f;
    const float actuator_Kp = 25.0f;
    const float actuator_Kd = 0.5;
    const double joint_vel_abs_limit = 30;     // rad/s
    const double joint_torque_abs_limit = 46;  // Nm
    // Only roll and pitch, does not make sense to limit yaw
    const std::array<std::pair<float, float>, 2> base_orientation_limit_rad{std::pair<float, float>{-0.6, 0.6}, {-0.6, 0.6}};
    // Isaac Lab joint order, rad
    const std::array<std::pair<float, float>, num_joints> joint_position_limits{std::pair<float, float>{-0.9, 0.9}, {-0.9, 0.9}, {-0.9, 0.9},
        {-0.9, 0.9}, {-1.4, 3.4}, {-1.4, 3.4}, {-1.4, 3.4}, {-1.4, 3.4}, {-3, -0.7}, {-3, -0.7}, {-3, -0.7}, {-3, -0.7}};
    std::array<float, num_joints> default_joint_positions{0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5};

private:
    void robot_state_callback(const unitree_go::msg::LowState::SharedPtr msg, const rclcpp::MessageInfo & message_info)
    {
        auto steady_now = std::chrono::steady_clock::now();
        auto system_now = std::chrono::system_clock::now();
        if (shutdown_coordinator_.handle_exit_if_requested() ||
            time_utils::shutdown_if_deadline_exceeded(last_state_callback_time_, std::chrono::milliseconds{50}, shutdown_coordinator_))
        {
            return;
        }

        // TODO: Remove because this was only used to check if /lowcmd works
        if (low_level_mode_enabled_.load(std::memory_order_acquire) && !initial_state_latched_.load(std::memory_order_acquire)) {
            initial_state_ = *msg;
            start_time_ = this->get_clock()->now().seconds();
            initial_state_latched_.store(true, std::memory_order_release);
            RCLCPP_INFO(this->get_logger(), "Latched. FR Calf: %f, FL Calf: %f", initial_state_.motor_state[2].q, initial_state_.motor_state[5].q);
        }

        // Backdate a local steady_clock rather than using the DDS system_clock directly because the latter are vulnerable to NTP time-jumps, which
        // can cause cause the message age check during policy inference to falsely pass. Using steady_clock guarantees monotonic age calculations.
        auto steady_publish_time = time_utils::get_safe_monotonic_publish_time(message_info, this->get_logger(), steady_now, system_now);
        auto stamped_state = stamped_state_from_lowstate(*msg, state_callback_iteration_counter_++, steady_publish_time);
        global_robot_state_.try_store_for(stamped_state, atomic_op_timeout_threshold);

        // State safety check
        // TODO: Use templates and functions to avoid repetition
        if (stamped_state.body_rpy_xyz[0] < base_orientation_limit_rad[0].first ||
            stamped_state.body_rpy_xyz[0] > base_orientation_limit_rad[0].second)
        {
            shutdown_coordinator_.shutdown(std::format("Base roll angle out of bounds, roll={}, bounds=[{},{}]", stamped_state.body_rpy_xyz[0],
                base_orientation_limit_rad[0].first, base_orientation_limit_rad[0].second));
        }

        if (stamped_state.body_rpy_xyz[1] < base_orientation_limit_rad[1].first ||
            stamped_state.body_rpy_xyz[1] > base_orientation_limit_rad[1].second)
        {
            shutdown_coordinator_.shutdown(std::format("Base pitch angle out of bounds, pitch={}, bounds=[{},{}]", stamped_state.body_rpy_xyz[1],
                base_orientation_limit_rad[1].first, base_orientation_limit_rad[1].second));
        }

        for (int i = 0; i < num_joints; i++) {
            if (stamped_state.joint_pos[i] < joint_position_limits[i].first || stamped_state.joint_pos[i] > joint_position_limits[i].second) {
                shutdown_coordinator_.shutdown(std::format("Joint position for index {} out of bounds, pos={}, bounds=[{},{}]", i,
                    stamped_state.joint_pos[i], joint_position_limits[i].first, joint_position_limits[i].second));
            }

            if (std::abs(stamped_state.joint_torque[i]) > joint_torque_abs_limit) {
                shutdown_coordinator_.shutdown(std::format(
                    "Joint torque for index {} out of bounds, torque={}, limit={}", i, stamped_state.joint_torque[i], joint_torque_abs_limit));
            }

            if (std::abs(stamped_state.joint_vel[i]) > joint_vel_abs_limit) {
                shutdown_coordinator_.shutdown(std::format(
                    "Joint velocity for index {} out of bounds, velocity={}, limit={}", i, stamped_state.joint_vel[i], joint_vel_abs_limit));
            }
        }
    }

    void policy_inference_callback()
    {
        if (shutdown_coordinator_.handle_exit_if_requested() ||
            time_utils::shutdown_if_deadline_exceeded(last_inference_callback_time_, std::chrono::milliseconds{30}, shutdown_coordinator_))
        {
            return;
        }
    }

    // Sends latest generated actions to the robot at steady 500Hz, as policy only runs at 50Hz.
    // This could also run in the state callback, but since these callbacks are run at 500Hz, it is important to keep them as lightweight as possible.
    // Another benefit is that network latency spikes for the received state cannot directly influence the actions being published, only after they
    // become large enough to trigger the stale state warning. Otherwise, the command timmer simply publishes actions based on latest state.
    // Because the stale state threshold does not allow extreme delays, this will "smooth out" temporary jitter
    void publish_commands()
    {
        if (shutdown_coordinator_.handle_exit_if_requested() ||
            time_utils::shutdown_if_deadline_exceeded(last_command_callback_time_, std::chrono::milliseconds{30}, shutdown_coordinator_))
        {
            return;
        }

        if (!low_level_mode_enabled_.load(std::memory_order_acquire)) {
            std::string error_message;
            const LowLevelModeEnabler::Status status = low_level_mode_enabler_.poll_thread_unsafe(error_message);

            if (status == LowLevelModeEnabler::Status::Failed) {
                shutdown_coordinator_.shutdown(error_message);
                return;
            }

            if (status == LowLevelModeEnabler::Status::Running) {
                RCLCPP_INFO_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000, "Waiting for motion switcher helper to release high-level control...");
                return;
            }

            if (status == LowLevelModeEnabler::Status::Succeeded) {
                low_level_mode_enabled_.store(true, std::memory_order_release);
                low_level_mode_enabled_time_ = this->get_clock()->now();
                RCLCPP_INFO(this->get_logger(), "Motion switcher helper exited successfully. Starting low-level control.");
            } else {
                return;
            }
        }

        const bool SINUSOIDAL_DEBUG_MOTION = true;
        if (SINUSOIDAL_DEBUG_MOTION) {
            if (!initial_state_latched_.load(std::memory_order_acquire)) {
                const double seconds_since_low_level_enabled = (this->get_clock()->now() - low_level_mode_enabled_time_).seconds();
                if (seconds_since_low_level_enabled > initial_state_latch_timeout_seconds_) {
                    shutdown_coordinator_.shutdown("Low-level mode was enabled, but initial /lowstate was not latched in time.");
                }
                return;
            }

            const double t = this->get_clock()->now().seconds() - start_time_;
            const double offset = 0.15 * (1.0 - std::cos(2.0 * M_PI * 0.25 * t));
            const int fr_calf = 2;
            const int fl_calf = 5;
            command_msg_.motor_cmd[fr_calf].q = initial_state_.motor_state[fr_calf].q + offset;
            command_msg_.motor_cmd[fr_calf].dq = 0.0;
            command_msg_.motor_cmd[fr_calf].kp = actuator_Kp;
            command_msg_.motor_cmd[fr_calf].kd = actuator_Kd;
            command_msg_.motor_cmd[fr_calf].tau = 0.0;
            command_msg_.motor_cmd[fl_calf].q = initial_state_.motor_state[fl_calf].q + offset;
            command_msg_.motor_cmd[fl_calf].dq = 0.0;
            command_msg_.motor_cmd[fl_calf].kp = actuator_Kp;
            command_msg_.motor_cmd[fl_calf].kd = actuator_Kd;
            command_msg_.motor_cmd[fl_calf].tau = 0.0;
        } else {
            auto setpoint_res = pd_setpoint_sdk_order.try_load_for(atomic_op_timeout_threshold);
            if (!setpoint_res.has_value()) {
                shutdown_coordinator_.shutdown(
                    std::format("Failed to fetch desired action within {}us in send_pd_commands(), exiting.", atomic_op_timeout_threshold.count()));
                return;
            }
            auto setpoint_sdk_order = setpoint_res.value();

            for (int i = 0; i < num_joints; i++) {
                command_msg_.motor_cmd[i].q = setpoint_sdk_order[i];
                command_msg_.motor_cmd[i].dq = 0;
                command_msg_.motor_cmd[i].kp = actuator_Kp;
                command_msg_.motor_cmd[i].kd = actuator_Kd;
                command_msg_.motor_cmd[i].tau = 0;
            }
        }

        get_crc(command_msg_);
        // Commented out for safety for now
        // if (shutdown_coordinator_.exit_requested()) {
        //     RCLCPP_WARN(this->get_logger(), "NOT publishing torque command because node shutdown was requested.");
        //     return;
        // }
        // command_publisher->publish(command_msg_);
    }

    // TODO: Move into history_buffer.hpp and rename to pytorch_helpers
    void load_pytorch_checkpoint()
    {
        // TODO: Make this ROS param
        // env 75, best one so far (with elevation map)
        const std::filesystem::path checkpoint_path{"/app/sim2real/traced_checkpoints/2025-06-28-17-13-04_21349_traced_deterministic.pt"};
        // env 75, same as above just ealier checkpoint to show before vs. after energy minimization std::filesystem::path checkpoint_path
        // std::filesystem::path checkpoint_path {"/app/sim2real/traced_checkpoints/2025-06-28-17-13-04_6049_traced_deterministic.pt"};

        // Safety precaution to ensure that trained policies do not receive significantly OOD observations.
        // To add new checkpoint, add to this list, hardcoded value will be set to zero automatically.
        // Otherwise, add to checkpoints_proper_elevation_map
        const std::vector<std::string> checkpoint_filenames_zero_elevation_map = {"2025-12-28-15-28-51_19499_traced_deterministic.pt",
            "2025-12-28-14-47-57_29499_traced_deterministic.pt", "2025-12-28-14-58-57_29649_traced_deterministic.pt"};
        const std::vector<std::string> checkpoint_filenames_proper_elevation_map = {
            "2025-06-28-17-13-04_21349_traced_deterministic.pt", "2025-06-28-17-13-04_6049_traced_deterministic.pt"};
        RCLCPP_INFO(this->get_logger(), "Checkpoints registered to use zeroed out hardcoded height map: %s",
            fmt::format("{}", fmt::join(checkpoint_filenames_zero_elevation_map, ", ")).c_str());
        RCLCPP_INFO(this->get_logger(), "Checkpoints registered to use proper elevation map: %s",
            fmt::format("{}", fmt::join(checkpoint_filenames_proper_elevation_map, ", ")).c_str());

        const bool checkpoint_in_zero_elevation_map =
            (std::find(checkpoint_filenames_zero_elevation_map.begin(), checkpoint_filenames_zero_elevation_map.end(),
                 checkpoint_path.filename().string()) != checkpoint_filenames_zero_elevation_map.end());
        const bool checkpoint_in_proper_elevation_map =
            (std::find(checkpoint_filenames_proper_elevation_map.begin(), checkpoint_filenames_proper_elevation_map.end(),
                 checkpoint_path.filename().string()) != checkpoint_filenames_proper_elevation_map.end());

        if (!checkpoint_in_zero_elevation_map && !checkpoint_in_proper_elevation_map) {
            shutdown_coordinator_.shutdown(
                "Specified checkpoint file found in neither of the two allowed checkpoint lists, exiting! This is a safety precaution to prevent "
                "passing incorrect observations into a policy, do not circumvent! Simply add the checkpoint to the correct list in the source code "
                "above this message printout.");
            return;
        }

        // TODO: Rework this to be a config file where each checkpoint is associated with certain configuration values
        if (checkpoint_in_zero_elevation_map) {
            RCLCPP_INFO(
                this->get_logger(), "Checkpoint found in zero elevation map list, setting use_hardcoded_heights=0 and hardcoded_elevation=0.0f");
            use_hardcoded_elevation = true;
            hardcoded_elevation = 0.0f;
        }  // Add adjustments for opposite scenario here if needed

        RCLCPP_INFO(this->get_logger(), "Loading torch policy checkpoint at path %s", checkpoint_path.string().c_str());
        try {
            policy_model_ = torch::jit::load(checkpoint_path.string());
            policy_model_.eval();
        } catch (const c10::Error & e) {
            shutdown_coordinator_.shutdown(std::format("Failed to load module, error message: {}", e.what()));
            return;
        }

        int64_t in_features = -1;
        for (const auto & p : policy_model_.named_parameters(/*recurse=*/true)) {
            if (p.name.ends_with(".weight") && p.value.dim() == 2) {
                in_features = p.value.size(1);
                break;
            }
        }

        if (in_features != observation_dim_no_history && in_features != observation_dim_history) {
            shutdown_coordinator_.shutdown(std::format("Observation dimension does not match expected value, exiting. in_features={}", in_features));
            return;
        }

        model_observation_dim_ = in_features;
        RCLCPP_INFO_STREAM(this->get_logger(),
            "Loaded module checkpoint from " << checkpoint_path.string() << "with observation dimension=" << model_observation_dim_);
    }

    // TODO: NOT FUNCTIONAL YET BECAUSE OF MISSING ELEVATION MAP PROCESSING!
    torch::Tensor construct_observation_tensor(const stamped_robot_state & robot_state, const std::array<float, 3> & vel_command,
        const std::array<float, num_joints> & previous_action, bool use_history, bool reset_history = false)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        const int obs_dim = use_history ? observation_dim_history : observation_dim_no_history;
        auto observation = torch::empty({1, obs_dim}, opts);

        auto base_ang_vel = torch::from_blob(const_cast<float *>(robot_state.body_angular_velocity.data()), {1, 3}, opts).clone();
        base_ang_vel.mul_(0.25f);
        observation.slice(1, 0, 3).copy_(base_ang_vel);

        auto velocity_cmd = torch::from_blob(const_cast<float *>(vel_command.data()), {1, 3}, opts).clone();
        velocity_cmd.mul_(torch::tensor({2.0f, 2.0f, 0.25f}, opts));
        observation.slice(1, 3, 6).copy_(velocity_cmd);

        auto projected_gravity = torch::from_blob(const_cast<float *>(robot_state.projected_gravity.data()), {1, 3}, opts).clone();
        projected_gravity.mul_(0.1f);
        observation.slice(1, 6, 9).copy_(projected_gravity);

        if (reset_history) {
            pos_hist_.reset();
            vel_hist_.reset();
        }

        auto jp_cur = torch::from_blob(const_cast<float *>(robot_state.joint_pos.data()), {num_joints}, opts).clone();
        auto jv_cur = torch::from_blob(const_cast<float *>(robot_state.joint_vel.data()), {num_joints}, opts).clone();
        jv_cur.mul_(0.05f);

        if (use_history) {
            pos_hist_.update(jp_cur);
            vel_hist_.update(jv_cur);

            const int pos_start = 9;
            const int vel_start = pos_start + history_length * num_joints;
            observation.slice(1, pos_start, pos_start + history_length * num_joints).copy_(pos_hist_.flattened().unsqueeze(0));
            observation.slice(1, vel_start, vel_start + history_length * num_joints).copy_(vel_hist_.flattened().unsqueeze(0));
        } else {
            observation.slice(1, 9, 9 + num_joints).copy_(jp_cur.unsqueeze(0));
            observation.slice(1, 21, 21 + num_joints).copy_(jv_cur.unsqueeze(0));
        }

        auto prev_action = torch::from_blob(const_cast<float *>(previous_action.data()), {1, num_joints}, opts).clone();
        const int prev_action_start_index = use_history ? (9 + 2 * history_length * num_joints) : (33);
        observation.slice(1, prev_action_start_index, prev_action_start_index + num_joints).copy_(prev_action);

        // const int height_map_start_index = prev_action_start_index + num_joints;
        // if (use_hardcoded_elevation) {
        //     observation.slice(1, height_map_start_index, obs_dim).fill_(hardcoded_elevation);
        // } else {
        //     auto elevation_data_result = global_elevation_map_filtered.try_load_for(std::chrono::microseconds(250));
        //     if (elevation_data_result.has_value()) {
        //         std::vector<float> map_vector = elevation_data_result.value();
        //         auto map_tensor_cpu = torch::from_blob(map_vector.data(), {1, elevation_grid_total_size},
        //             opts);  // Creates view but lives until end of scope, so copy in next line is sufficient
        //         observation.slice(1, height_map_start_index, height_map_start_index + elevation_grid_total_size).copy_(map_tensor_cpu);
        //     } else {
        //         shutdown_coordinator_.shutdown("Critical: Failed to load global elevation map in observation construction, exiting.");
        //     }
        // }

        return observation;
    }

    long long inference_iteration_counter_;
    long long state_callback_iteration_counter_;
    std::chrono::steady_clock::time_point last_state_callback_time_{};      // default = epoch
    std::chrono::steady_clock::time_point last_inference_callback_time_{};  // default = epoch
    std::chrono::steady_clock::time_point last_command_callback_time_{};    // default = epoch
    int model_observation_dim_;

    // TODO: Clean up once motion test is removed in favor of proper policy inference
    const std::string network_interface_;
    const double initial_state_latch_timeout_seconds_ = 2.0;
    rclcpp::Time low_level_mode_enabled_time_{0, 0, RCL_ROS_TIME};
    std::atomic<bool> initial_state_latched_{false};
    std::atomic<bool> low_level_mode_enabled_{false};
    double start_time_ = 0.0;
    unitree_go::msg::LowState initial_state_;

    unitree_go::msg::LowCmd command_msg_;
    timed_atomic<stamped_robot_state> global_robot_state_{};
    torch::jit::Module policy_model_;
    HistoryBuffer pos_hist_{history_length, num_joints, torch::kCPU};
    HistoryBuffer vel_hist_{history_length, num_joints, torch::kCPU};
    LowLevelModeEnabler low_level_mode_enabler_;
    ShutdownCoordinator shutdown_coordinator_;

    // Benefit of having a separate command timer compared to also publishing in the subscriber callback is that separation means network jitter does
    // not affect the commands as much
    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::TimerBase::SharedPtr policy_inference_timer_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr command_publisher;
    // TODO: Add subscriber for elevation map
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    if (argc < 2) {
        std::cerr << "No network interface specified, usage: " << argv[0] << " [networkInterface]" << std::endl;
        return EXIT_FAILURE;
    }

    rclcpp::spin(std::make_shared<CaTControlNode>(argv[1]));
    rclcpp::shutdown();
    return 0;
}