#include <fmt/core.h>
#include <fmt/ranges.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "cat_controller/inference_engine.hpp"
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
    explicit CaTControlNode(const std::string & network_interface, const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
        : Node("cat_control_node", options),
          network_interface_(network_interface),  // TODO: Make this a ros param
          use_hardcoded_elevation_(declare_use_hardcoded_elevation()),
          checkpoint_path_(declare_checkpoint_path()),
          inference_engine_(checkpoint_path_, num_joints),
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

        if (use_hardcoded_elevation_) {
            RCLCPP_INFO(this->get_logger(), "use_hardcoded_elevation=true, setting hardcoded_elevation to zero!");
            hardcoded_elevation_ = 0.0f;
        }
        RCLCPP_INFO(this->get_logger(), "checkpoint_path='%s', use_hardcoded_elevation=%s", checkpoint_path_.string().c_str(),
            use_hardcoded_elevation_ ? "true" : "false");

        RCLCPP_DEBUG(this->get_logger(), "Starting robot state subscriber.");
        robot_state_sub_ = this->create_subscription<unitree_go::msg::LowState>("/lowstate", rclcpp::SensorDataQoS(),
            std::bind(&CaTControlNode::robot_state_callback, this, std::placeholders::_1, std::placeholders::_2));
        RCLCPP_DEBUG(this->get_logger(), "Started robot state subscriber.");

        RCLCPP_DEBUG(this->get_logger(), "Starting robot command publisher.");
        command_publisher_ = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", rclcpp::SensorDataQoS());
        RCLCPP_DEBUG(this->get_logger(), "Started robot command publisher.");

        std::string error_message;
        RCLCPP_DEBUG(this->get_logger(), "Starting low level control mode enabler process.");
        if (!low_level_mode_enabler_.start(error_message)) {
            // Throw here because we have not initialized anything that warrants proper shutdown. After the constructor, we use the shutdown
            // coordinator to avoid exceptions
            throw std::runtime_error(std::format("Failed to enable low level control mode, throwing. Error message: {}", error_message));
        }
        RCLCPP_INFO(this->get_logger(), "Started motion switcher helper using interface '%s'.", network_interface_.c_str());

        RCLCPP_DEBUG(this->get_logger(), "Starting robot command publish timer.");
        command_timer_ = this->create_wall_timer(2ms, std::bind(&CaTControlNode::publish_commands, this));
        RCLCPP_DEBUG(this->get_logger(), "Started robot command publish timer.");

        RCLCPP_DEBUG(this->get_logger(), "Starting policy inference / control loop timer.");
        policy_inference_timer_ = this->create_wall_timer(20ms, std::bind(&CaTControlNode::policy_inference_callback, this));
        RCLCPP_DEBUG(this->get_logger(), "Started policy inference / control loop timer.");
    }

    std::chrono::microseconds atomic_op_timeout_threshold{500};
    std::chrono::milliseconds stale_state_age_threshold{50};
    const bool walk_a_bit = true;
    const static short num_joints = 12;
    const float action_scale = 0.8f;
    const float actuator_Kp = 25.0f;
    const float actuator_Kd = 0.5f;
    const double joint_vel_abs_limit = 30.0f;                      // rad/s
    const double joint_torque_abs_limit = 46.0f;                   // Nm
    std::array<float, 3> vel_command_mag_limit = {2.0, 2.0, 1.0};  // vel_x, vel_y, omega_z
    // Only roll and pitch, does not make sense to limit yaw
    const std::array<std::pair<float, float>, 2> base_orientation_limit_rad{std::pair<float, float>{-0.6, 0.6}, {-0.6, 0.6}};
    // Isaac Lab joint order, rad
    const std::array<std::pair<float, float>, num_joints> joint_position_limits{std::pair<float, float>{-0.9, 0.9}, {-0.9, 0.9}, {-0.9, 0.9},
        {-0.9, 0.9}, {-1.4, 3.4}, {-1.4, 3.4}, {-1.4, 3.4}, {-1.4, 3.4}, {-3, -0.7}, {-3, -0.7}, {-3, -0.7}, {-3, -0.7}};
    std::array<float, num_joints> default_joint_positions_isaac_order{0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5};

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

        // This code path is only used when SINUSOIDAL_DEBUG_MOTION is true in the command publisher, used to ensure that /lowcmd messages being
        // published are actually being applied.
        if (low_level_mode_enabled_.load(std::memory_order_acquire) && !initial_low_level_state_.load(std::memory_order_acquire)) {
            initial_state_ = *msg;
            start_time_ = this->get_clock()->now().seconds();
            initial_low_level_state_.store(true, std::memory_order_release);
            RCLCPP_INFO(this->get_logger(), "Low level mode enabled, saving initial state. FR Calf: %f, FL Calf: %f", initial_state_.motor_state[2].q,
                initial_state_.motor_state[5].q);
            start_ms_policy_inference_ =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
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
        if (shutdown_coordinator_.handle_exit_if_requested()) { return; }
        // Skip the deadline check for the first few iterations due to PyTorch warmup
        if (inference_iteration_counter_ <= 5) {
            last_inference_callback_time_ = std::chrono::steady_clock::now();
        } else if (time_utils::shutdown_if_deadline_exceeded(last_inference_callback_time_, std::chrono::milliseconds{30}, shutdown_coordinator_)) {
            return;
        }

        auto robot_state_res = global_robot_state_.try_load_for(atomic_op_timeout_threshold);
        if (!robot_state_res.has_value()) {
            shutdown_coordinator_.shutdown(std::format("Failed to retrieve robot state within {}us, exiting.", atomic_op_timeout_threshold.count()));
            return;
        }
        auto robot_state = robot_state_res.value();
        auto now = std::chrono::steady_clock::now();
        auto delta = now - robot_state.timestamp;
        if (delta > stale_state_age_threshold && robot_state.counter > 0) {  // Discard first iteration
            shutdown_coordinator_.shutdown(
                std::format("State timestamp too old, allowed threshold={}ms, actual state age={}ms. Exiting to prevent outdated states.",
                    stale_state_age_threshold.count(), std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()));
            return;
        }

        std::array<float, 3> vel_command{0.0, 0.0, 0.0};
        if (auto vcmd = global_vel_command.try_load_for(atomic_op_timeout_threshold); vcmd.has_value()) {
            vel_command = vcmd.value();
        } else {
            shutdown_coordinator_.shutdown(std::format("Failed to fetch vel_command within {}us, exiting.", atomic_op_timeout_threshold.count()));
            return;
        }

        // Clip velocity command components
        for (int i = 0; i < 3; i++) {
            if (std::abs(vel_command[i]) > vel_command_mag_limit[i]) {
                vel_command[i] = std::max(-vel_command_mag_limit[i], std::min(vel_command_mag_limit[i], vel_command[i]));
                RCLCPP_WARN_STREAM(this->get_logger(),
                    "Had to clip vel_command[" << i << "]=" << vel_command[i] << ", vel_command_mag_limit[" << i << "]=" << vel_command_mag_limit[i]);
            }
        }

        auto time_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        auto rel_time_ms = time_now_ms - start_ms_policy_inference_;
        if (walk_a_bit && rel_time_ms > 30000 && rel_time_ms < 34500) { vel_command[0] = 0.9f; }
        auto generated_action = inference_engine_.generate_action(robot_state, vel_command);
        // Do not check if target exceeds joint limits because policy might learn to command out of range values temporarily for more rapid motion.

        std::array<float, num_joints> pd_target_sdk_order{};  // Go2 SDK native order, NOT Isaac Lab!!!
        for (int i = 0; i < num_joints; i++) {
            int j = sdk_to_isaac_idx[i];                                                                           // Remap to go2 order
            pd_target_sdk_order[i] = default_joint_positions_isaac_order[j] + generated_action[j] * action_scale;  // Scale same as Isaac Lab
        }
        if (!pd_setpoint_sdk_order.try_store_for(pd_target_sdk_order, atomic_op_timeout_threshold)) {
            shutdown_coordinator_.shutdown(
                std::format("Failed to update global PD target within {}us, exiting.", atomic_op_timeout_threshold.count()));
            return;
        }

        inference_iteration_counter_++;
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

        if (!initial_low_level_state_.load(std::memory_order_acquire)) {
            const double seconds_since_low_level_enabled = (this->get_clock()->now() - low_level_mode_enabled_time_).seconds();
            if (seconds_since_low_level_enabled > initial_state_save_timeout_seconds_) {
                shutdown_coordinator_.shutdown("Low-level mode was enabled, but initial /lowstate was not saved in time.");
            }
            return;
        }

        // Compute relative time since low level control mode was enabled and initial state was captured
        const double t = this->get_clock()->now().seconds() - start_time_;
        const double interpolation_duration = 5.0f;
        std::array<float, num_joints> current_target_sdk{};
        const bool SINUSOIDAL_DEBUG_MOTION = true;

        if (t < interpolation_duration) {
            // Interpolate from initial state to default standing position
            for (int i = 0; i < num_joints; i++) {
                int j = sdk_to_isaac_idx[i];
                float default_pos = default_joint_positions_isaac_order[j];
                float initial_pos = initial_state_.motor_state[i].q;
                current_target_sdk[i] = (t / interpolation_duration) * default_pos + (1.0 - (t / interpolation_duration)) * initial_pos;
            }
            // Set to latest state so that no jumps occur when switching to applying this setpoint
            pd_setpoint_sdk_order.try_store_for(current_target_sdk, atomic_op_timeout_threshold);

        } else {
            if (!interpolation_finished_.load(std::memory_order_acquire)) {
                interpolation_finished_.store(true, std::memory_order_release);
                RCLCPP_INFO(this->get_logger(), "Interpolation finished, active control started.");
            }

            if (SINUSOIDAL_DEBUG_MOTION) {
                // Shift time so the sine wave starts smoothly at t=0 relative to interpolation end
                const double debug_t = t - interpolation_duration;
                const double offset = 0.1 * (1.0 - std::cos(2.0 * M_PI * 0.25 * debug_t));
                const int fr_calf = 2;
                const int fl_calf = 5;

                // Base everything on the standing pose
                for (int i = 0; i < num_joints; i++) { current_target_sdk[i] = default_joint_positions_isaac_order[sdk_to_isaac_idx[i]]; }
                current_target_sdk[fr_calf] += offset;
                current_target_sdk[fl_calf] += offset;

            } else {
                auto setpoint_res = pd_setpoint_sdk_order.try_load_for(atomic_op_timeout_threshold);
                if (!setpoint_res.has_value()) {
                    shutdown_coordinator_.shutdown(
                        std::format("Failed to fetch desired action within {}us, exiting.", atomic_op_timeout_threshold.count()));
                    return;
                }
                current_target_sdk = setpoint_res.value();
            }
        }

        for (int i = 0; i < num_joints; i++) {
            command_msg_.motor_cmd[i].q = current_target_sdk[i];
            command_msg_.motor_cmd[i].dq = 0.0;
            command_msg_.motor_cmd[i].kp = actuator_Kp;
            command_msg_.motor_cmd[i].kd = actuator_Kd;
            command_msg_.motor_cmd[i].tau = 0.0;
        }
        get_crc(command_msg_);
        // Commented out for safety for now
        // if (shutdown_coordinator_.exit_requested()) {
        //     RCLCPP_WARN(this->get_logger(), "NOT publishing torque command because node shutdown was requested.");
        //     return;
        // }
        // command_publisher_->publish(command_msg_);
    }

    bool declare_use_hardcoded_elevation() { return this->declare_parameter<bool>("use_hardcoded_elevation"); }
    std::filesystem::path declare_checkpoint_path()
    {
        const std::string checkpoint_path_str = this->declare_parameter<std::string>("checkpoint_path");
        const std::filesystem::path checkpoint_path{checkpoint_path_str};

        if (!std::filesystem::exists(checkpoint_path)) {
            throw std::runtime_error(std::format("checkpoint_path={} does not exist, throwing.", checkpoint_path.string()));
        }
        return checkpoint_path;
    }

    // REMEMBER THAT ORDER MATTERS HERE FOR INITIALIZATION
    // This is why checkpoint_path and use_hardcoded_elevation are at the top, it is necessary to init them using the parsed ROS params for the
    // inference engine to initialize correctly!
    const std::string network_interface_;
    const bool use_hardcoded_elevation_;
    double hardcoded_elevation_ = -0.3f;
    const std::filesystem::path checkpoint_path_;
    long long inference_iteration_counter_{};
    long long state_callback_iteration_counter_{};
    int64_t start_ms_policy_inference_{};
    std::chrono::steady_clock::time_point last_state_callback_time_{};      // default = epoch
    std::chrono::steady_clock::time_point last_inference_callback_time_{};  // default = epoch
    std::chrono::steady_clock::time_point last_command_callback_time_{};    // default = epoch

    const double initial_state_save_timeout_seconds_{2.0};
    rclcpp::Time low_level_mode_enabled_time_{0, 0, RCL_ROS_TIME};
    std::atomic<bool> initial_low_level_state_{false};  // Robot state when low level mode was enabled
    std::atomic<bool> low_level_mode_enabled_{false};
    std::atomic<bool> interpolation_finished_{false};
    double start_time_{0};
    unitree_go::msg::LowState initial_state_;

    unitree_go::msg::LowCmd command_msg_;
    timed_atomic<stamped_robot_state> global_robot_state_{};
    timed_atomic<std::array<float, num_joints>> pd_setpoint_sdk_order{};
    timed_atomic<std::array<float, 3>> global_vel_command{{0.0f, 0.0f, 0.0f}};

    InferenceEngine inference_engine_;
    LowLevelModeEnabler low_level_mode_enabler_;
    ShutdownCoordinator shutdown_coordinator_;

    // Benefit of having a separate command timer compared to also publishing in the subscriber callback
    // is that separation means network jitter does not affect the commands as much
    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::TimerBase::SharedPtr policy_inference_timer_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr command_publisher_;
    // TODO: Add subscriber for PROCESSED elevation map (separate node will handle making it robot-centric so that this node just needs to pass array
    // of floats to InferenceEngine)
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