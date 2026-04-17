#include <fmt/core.h>
#include <fmt/ranges.h>

#include <ament_index_cpp/get_package_prefix.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <optional>
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
    explicit CaTControlNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
        : Node("cat_control_node", options),
          network_interface_(declare_and_get_param<std::string>("network_interface", "Network interface for Go2", true)),
          use_hardcoded_elevation_(declare_and_get_param<bool>("use_hardcoded_elevation", "Override elevation map", true)),
          hardcoded_elevation_(declare_and_get_param<double>("hardcoded_elevation", "Elevation value if hardcoded is true")),
          checkpoint_path_str_(declare_and_get_param<std::string>("checkpoint_path", "Path to PyTorch model", true)),

          atomic_op_timeout_threshold_(std::chrono::microseconds(declare_and_get_param<int>("atomic_op_timeout_us"))),
          stale_state_age_threshold_(std::chrono::milliseconds(declare_and_get_param<int>("stale_state_age_ms"))),

          walk_a_bit_(declare_and_get_param<bool>("walk_a_bit", "Enable debug walk sequence")),
          sinusoidal_debug_motion_(declare_and_get_param<bool>("sinusoidal_debug_motion", "Enable sinusoidal testing motion")),

          action_scale_(declare_and_get_param<double>("action_scale")),
          actuator_Kp_(declare_and_get_param<double>("actuator_kp")),
          actuator_Kd_(declare_and_get_param<double>("actuator_kd")),
          joint_vel_abs_limit_(declare_and_get_param<double>("joint_vel_abs_limit")),
          joint_torque_abs_limit_(declare_and_get_param<double>("joint_torque_abs_limit")),
          vel_command_mag_limit_(declare_and_get_param<std::vector<double>>("vel_command_mag_limit")),

          interpolation_duration_(declare_and_get_param<double>("interpolation_duration_sec")),
          initial_state_save_timeout_seconds_(declare_and_get_param<double>("initial_state_save_timeout_sec")),

          checkpoint_path_(validate_checkpoint_path(checkpoint_path_str_)),
          inference_engine_(checkpoint_path_, NUM_JOINTS),
          low_level_mode_enabler_(
              ament_index_cpp::get_package_prefix("cat_controller") + "/lib/cat_controller/release_motion_mode", network_interface_, 45.0),

          shutdown_coordinator_(this->get_logger(), this->get_node_base_interface()->get_context(), [this]() {
              // Very important to put any cleanup for the node here!
              if (command_timer_) { command_timer_->cancel(); }
              if (policy_inference_timer_) { policy_inference_timer_->cancel(); }
              low_level_mode_enabler_.stop();
          })
    {
        static_assert(std::atomic<bool>::is_always_lock_free, "atomic bool is not lock free.");
        init_command_msg(command_msg_);

        this->state_sub_cbg_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        this->command_timer_cbg_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        this->inference_timer_cbg_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        if (use_hardcoded_elevation_) {
            RCLCPP_INFO(this->get_logger(), "use_hardcoded_elevation=true, setting hardcoded_elevation to zero!");
            hardcoded_elevation_ = 0.0f;
        }
        RCLCPP_INFO(this->get_logger(), "checkpoint_path='%s', use_hardcoded_elevation=%s", checkpoint_path_.string().c_str(),
            use_hardcoded_elevation_ ? "true" : "false");

        RCLCPP_DEBUG(this->get_logger(), "Starting robot state subscriber.");
        rclcpp::SubscriptionOptions state_sub_options;
        state_sub_options.callback_group = state_sub_cbg_;
        robot_state_sub_ = this->create_subscription<unitree_go::msg::LowState>(
            "/lowstate", rclcpp::SensorDataQoS(), std::bind(&CaTControlNode::robot_state_callback, this, std::placeholders::_1), state_sub_options);
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
        command_timer_ = this->create_wall_timer(2ms, std::bind(&CaTControlNode::publish_commands, this), command_timer_cbg_);
        RCLCPP_DEBUG(this->get_logger(), "Started robot command publish timer.");

        RCLCPP_DEBUG(this->get_logger(), "Starting policy inference / control loop timer.");
        policy_inference_timer_ = this->create_wall_timer(20ms, std::bind(&CaTControlNode::policy_inference_callback, this), inference_timer_cbg_);
        RCLCPP_DEBUG(this->get_logger(), "Started policy inference / control loop timer.");

        // Dump all node parameters to logs
        std::string param_dump = "=== Node Parameters ===\n";
        auto param_list = this->list_parameters(std::vector<std::string>{}, 10);
        for (const auto & param_name : param_list.names) {
            rclcpp::Parameter param;
            if (this->get_parameter(param_name, param)) { param_dump += fmt::format("  {}: {}\n", param_name, param.value_to_string()); }
        }
        param_dump += "============================";
        RCLCPP_INFO(this->get_logger(), "\n%s", param_dump.c_str());
    }

private:
    static constexpr short NUM_JOINTS = 12;
    // Arrays of pairs are not supported by ROS parameters and hardware bounds shouldn't change at runtime anyway
    static constexpr std::array<std::pair<float, float>, 2> BASE_ORIENTATION_LIMIT_RAD = {std::pair<float, float>{-0.6f, 0.6f}, {-0.6f, 0.6f}};
    static constexpr std::array<std::pair<float, float>, NUM_JOINTS> JOINT_POSITION_LIMITS = {std::pair<float, float>{-0.9f, 0.9f}, {-0.9f, 0.9f},
        {-0.9f, 0.9f}, {-0.9f, 0.9f}, {-1.4f, 3.4f}, {-1.4f, 3.4f}, {-1.4f, 3.4f}, {-1.4f, 3.4f}, {-3.0f, -0.7f}, {-3.0f, -0.7f}, {-3.0f, -0.7f},
        {-3.0f, -0.7f}};
    static constexpr std::array<float, NUM_JOINTS> DEFAULT_JOINT_POSITIONS_ISAAC_ORDER = {
        0.1f, -0.1f, 0.1f, -0.1f, 0.8f, 0.8f, 1.0f, 1.0f, -1.5f, -1.5f, -1.5f, -1.5f};

    template <typename T>
    [[nodiscard]] constexpr bool is_out_of_bounds(const T & val, const std::pair<T, T> & bounds)
    { return val < bounds.first || val > bounds.second; }

    std::optional<std::string> validate_robot_state(const stamped_robot_state & state)
    {
        const std::array<std::string, 2> rpy_names = {"roll", "pitch"};
        for (size_t i = 0; i < 2; ++i) {
            if (is_out_of_bounds(state.body_rpy_xyz[i], BASE_ORIENTATION_LIMIT_RAD[i])) {
                return std::format("Base {} angle out of bounds, value={}, bounds=[{},{}]", rpy_names[i], state.body_rpy_xyz[i],
                    BASE_ORIENTATION_LIMIT_RAD[i].first, BASE_ORIENTATION_LIMIT_RAD[i].second);
            }
        }

        for (int i = 0; i < NUM_JOINTS; i++) {
            if (is_out_of_bounds(state.joint_pos[i], JOINT_POSITION_LIMITS[i])) {
                return std::format("Joint position for index {} out of bounds, pos={}, bounds=[{},{}]", i, state.joint_pos[i],
                    JOINT_POSITION_LIMITS[i].first, JOINT_POSITION_LIMITS[i].second);
            }

            if (std::abs(state.joint_torque[i]) > joint_torque_abs_limit_) {
                return std::format("Joint torque for index {} out of bounds, torque={}, limit={}", i, state.joint_torque[i], joint_torque_abs_limit_);
            }

            if (std::abs(state.joint_vel[i]) > joint_vel_abs_limit_) {
                return std::format("Joint velocity for index {} out of bounds, velocity={}, limit={}", i, state.joint_vel[i], joint_vel_abs_limit_);
            }
        }

        return std::nullopt;  // State is valid and safe
    }

    template <typename T>
    T declare_and_get_param(const std::string & name, const std::string & description = "", bool read_only = false)
    {
        rcl_interfaces::msg::ParameterDescriptor desc;
        desc.description = description;
        desc.read_only = read_only;

        // We can again throw here because it only happens during constructor of node
        try {
            this->declare_parameter<T>(name, desc);  // Declare without default to require specifying a value
            return this->get_parameter(name).get_value<T>();
        } catch (const rclcpp::exceptions::ParameterUninitializedException & e) {
            RCLCPP_FATAL(this->get_logger(), "CRITICAL: Mandatory parameter '%s' is missing from config!", name.c_str());
            throw;
        } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
            RCLCPP_FATAL(this->get_logger(), "CRITICAL: Parameter '%s' has the wrong type in YAML!", name.c_str());
            throw;
        }
    }

    std::filesystem::path validate_checkpoint_path(const std::string & path_str)
    {
        std::filesystem::path path{path_str};
        if (!std::filesystem::exists(path)) { throw std::runtime_error(std::format("checkpoint_path={} does not exist.", path.string())); }
        return path;
    }

    void robot_state_callback(const unitree_go::msg::LowState::SharedPtr msg)
    {
        auto steady_now = std::chrono::steady_clock::now();
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

        // Use local receive time instead of DDS source timestamps to prevent false-positive negative message ages clock skew on the Go2.
        // Downside is that we assume message age is zero when it arrives but there is no other way because LowState does not include a timestamp
        auto stamped_state = stamped_state_from_lowstate(*msg, state_callback_iteration_counter_++, steady_now);

        if (auto error_message = validate_robot_state(stamped_state)) {
            shutdown_coordinator_.shutdown(*error_message);
            return;
        }
        // Only store if valid
        global_robot_state_.try_store_for(stamped_state, atomic_op_timeout_threshold_);
    }

    void policy_inference_callback()
    {
        if (shutdown_coordinator_.handle_exit_if_requested() || !interpolation_finished_.load(std::memory_order::acquire)) { return; }
        // Skip the deadline check for the first few iterations due to PyTorch warmup
        if (inference_iteration_counter_ <= 5) {
            last_inference_callback_time_ = std::chrono::steady_clock::now();
        } else if (time_utils::shutdown_if_deadline_exceeded(last_inference_callback_time_, std::chrono::milliseconds{30}, shutdown_coordinator_)) {
            return;
        }

        auto robot_state_res = global_robot_state_.try_load_for(atomic_op_timeout_threshold_);
        if (!robot_state_res.has_value()) {
            shutdown_coordinator_.shutdown(std::format("Failed to retrieve robot state within {}us, exiting.", atomic_op_timeout_threshold_.count()));
            return;
        }
        auto robot_state = robot_state_res.value();
        auto now = std::chrono::steady_clock::now();
        auto delta = now - robot_state.timestamp;
        if (delta > stale_state_age_threshold_ && robot_state.counter > 0) {  // Discard first iteration
            shutdown_coordinator_.shutdown(
                std::format("State timestamp too old, allowed threshold={}ms, actual state age={}ms. Exiting to prevent outdated states.",
                    stale_state_age_threshold_.count(), std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()));
            return;
        }

        std::array<float, 3> vel_command{0.0, 0.0, 0.0};
        if (auto vcmd = global_vel_command.try_load_for(atomic_op_timeout_threshold_); vcmd.has_value()) {
            vel_command = vcmd.value();
        } else {
            shutdown_coordinator_.shutdown(std::format("Failed to fetch vel_command within {}us, exiting.", atomic_op_timeout_threshold_.count()));
            return;
        }

        // Clip velocity command components
        for (int i = 0; i < 3; i++) {
            // Cast the double parameter to a float for comparison because ROS2 only supports double for parameter parsing
            const float limit = static_cast<float>(vel_command_mag_limit_[i]);
            if (std::abs(vel_command[i]) > limit) {
                vel_command[i] = std::clamp(vel_command[i], -limit, limit);
                RCLCPP_WARN_STREAM(this->get_logger(),
                    "Had to clip vel_command[" << i << "]=" << vel_command[i] << ", vel_command_mag_limit[" << i << "]=" << limit);
            }
        }

        auto time_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        auto rel_time_ms = time_now_ms - start_ms_policy_inference_;
        if (walk_a_bit_ && rel_time_ms > 30000 && rel_time_ms < 34500) { vel_command[0] = 0.9f; }
        const auto & generated_action = inference_engine_.generate_action(robot_state, vel_command);
        // Do not check if target exceeds joint limits because policy might learn to command out of range values temporarily for more rapid
        // motion.

        std::array<float, NUM_JOINTS> pd_target_sdk_order{};  // Go2 SDK native order, NOT Isaac Lab!!!
        for (int i = 0; i < NUM_JOINTS; i++) {
            int j = sdk_to_isaac_idx[i];                                                                            // Remap to go2 order
            pd_target_sdk_order[i] = DEFAULT_JOINT_POSITIONS_ISAAC_ORDER[j] + generated_action[j] * action_scale_;  // Scale same as Isaac Lab
        }
        if (!pd_setpoint_sdk_order.try_store_for(pd_target_sdk_order, atomic_op_timeout_threshold_)) {
            shutdown_coordinator_.shutdown(
                std::format("Failed to update global PD target within {}us, exiting.", atomic_op_timeout_threshold_.count()));
            return;
        }

        inference_iteration_counter_++;
    }

    // Sends latest generated actions to the robot at steady 500Hz, as policy only runs at 50Hz.
    // This could also run in the state callback, but since these callbacks are run at 500Hz, it is important to keep them as lightweight as
    // possible. Another benefit is that network latency spikes for the received state cannot directly influence the actions being published, only
    // after they become large enough to trigger the stale state warning. Otherwise, the command timmer simply publishes actions based on latest
    // state. Because the stale state threshold does not allow extreme delays, this will "smooth out" temporary jitter
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
        std::array<float, NUM_JOINTS> current_target_sdk{};
        const bool SINUSOIDAL_DEBUG_MOTION = true;

        if (t < interpolation_duration) {
            // Interpolate from initial state to default standing position
            const float ratio = std::clamp(static_cast<float>(t / interpolation_duration), 0.0f, 1.0f);
            for (int i = 0; i < NUM_JOINTS; i++) {
                int j = sdk_to_isaac_idx[i];
                float default_pos = DEFAULT_JOINT_POSITIONS_ISAAC_ORDER[j];
                float initial_pos = initial_state_.motor_state[i].q;
                current_target_sdk[i] = ratio * default_pos + (1.0f - ratio) * initial_pos;
            }
            // Set to latest state so that no jumps occur when switching to applying this setpoint
            pd_setpoint_sdk_order.try_store_for(current_target_sdk, atomic_op_timeout_threshold_);

        } else {
            if (!interpolation_finished_.load(std::memory_order_acquire)) {
                interpolation_finished_.store(true, std::memory_order_release);
                // Update setpoint with the final standing pose while we wait for the inference thread's first action
                pd_setpoint_sdk_order.try_store_for(current_target_sdk, atomic_op_timeout_threshold_);
                RCLCPP_INFO(this->get_logger(), "Interpolation finished, active control started.");
            }

            if (SINUSOIDAL_DEBUG_MOTION) {
                // Shift time so the sine wave starts smoothly at t=0 relative to interpolation end
                const double debug_t = t - interpolation_duration;
                const double offset = 0.2 * (1.0 - std::cos(2.0 * M_PI * 0.25 * debug_t));
                const int fr_calf = 2;
                const int fl_calf = 5;

                // Base everything on the standing pose
                for (int i = 0; i < NUM_JOINTS; i++) { current_target_sdk[i] = DEFAULT_JOINT_POSITIONS_ISAAC_ORDER[sdk_to_isaac_idx[i]]; }
                current_target_sdk[fr_calf] += offset;
                current_target_sdk[fl_calf] += offset;

            } else {
                auto setpoint_res = pd_setpoint_sdk_order.try_load_for(atomic_op_timeout_threshold_);
                if (!setpoint_res.has_value()) {
                    shutdown_coordinator_.shutdown(
                        std::format("Failed to fetch desired action within {}us, exiting.", atomic_op_timeout_threshold_.count()));
                    return;
                }
                current_target_sdk = setpoint_res.value();
            }
        }

        for (int i = 0; i < NUM_JOINTS; i++) {
            command_msg_.motor_cmd[i].q = current_target_sdk[i];
            command_msg_.motor_cmd[i].dq = 0.0;
            command_msg_.motor_cmd[i].kp = actuator_Kp_;
            command_msg_.motor_cmd[i].kd = actuator_Kd_;
            command_msg_.motor_cmd[i].tau = 0.0;
        }
        get_crc(command_msg_);
        // Commented out for safety for now
        if (shutdown_coordinator_.exit_requested()) {
            RCLCPP_WARN(this->get_logger(), "NOT publishing torque command because node shutdown was requested.");
            return;
        }
        command_publisher_->publish(command_msg_);
    }

    // REMEMBER THAT ORDER MATTERS HERE FOR INITIALIZATION
    // This is why checkpoint_path and use_hardcoded_elevation are at the top, it is necessary to init them using the parsed ROS params for the
    // inference engine to initialize correctly!
    const std::string network_interface_;
    const bool use_hardcoded_elevation_;
    double hardcoded_elevation_;
    const std::string checkpoint_path_str_;

    const std::chrono::microseconds atomic_op_timeout_threshold_;
    const std::chrono::milliseconds stale_state_age_threshold_;

    const bool walk_a_bit_;
    const bool sinusoidal_debug_motion_;

    const double action_scale_;
    const double actuator_Kp_;
    const double actuator_Kd_;
    const double joint_vel_abs_limit_;
    const double joint_torque_abs_limit_;
    const std::vector<double> vel_command_mag_limit_;

    const double interpolation_duration_;
    const double initial_state_save_timeout_seconds_;

    const std::filesystem::path checkpoint_path_;
    long long inference_iteration_counter_{0};
    long long state_callback_iteration_counter_{0};
    int64_t start_ms_policy_inference_{0};

    std::chrono::steady_clock::time_point last_state_callback_time_{};
    std::chrono::steady_clock::time_point last_inference_callback_time_{};
    std::chrono::steady_clock::time_point last_command_callback_time_{};

    rclcpp::Time low_level_mode_enabled_time_{0, 0, RCL_ROS_TIME};
    std::atomic<bool> initial_low_level_state_{false};
    std::atomic<bool> low_level_mode_enabled_{false};
    std::atomic<bool> interpolation_finished_{false};
    double start_time_{0.0};

    unitree_go::msg::LowState initial_state_;
    unitree_go::msg::LowCmd command_msg_;

    timed_atomic<stamped_robot_state> global_robot_state_{};
    timed_atomic<std::array<float, NUM_JOINTS>> pd_setpoint_sdk_order{};
    timed_atomic<std::array<float, 3>> global_vel_command{{0.0f, 0.0f, 0.0f}};

    InferenceEngine inference_engine_;
    LowLevelModeEnabler low_level_mode_enabler_;
    ShutdownCoordinator shutdown_coordinator_;

    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::TimerBase::SharedPtr policy_inference_timer_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr command_publisher_;

    rclcpp::CallbackGroup::SharedPtr state_sub_cbg_;
    rclcpp::CallbackGroup::SharedPtr command_timer_cbg_;
    rclcpp::CallbackGroup::SharedPtr inference_timer_cbg_;
    // TODO: Add subscriber for PROCESSED elevation map (separate node will handle making it robot-centric so that this node just needs to pass
    // array of floats to InferenceEngine)
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CaTControlNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}