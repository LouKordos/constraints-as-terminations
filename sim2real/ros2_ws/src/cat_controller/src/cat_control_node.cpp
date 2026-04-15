#include <fmt/core.h>
#include <fmt/ranges.h>
#include <torch/script.h>
#include <torch/torch.h>

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
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

using namespace std::chrono_literals;

class CaTControlNode : public rclcpp::Node
{
public:
    explicit CaTControlNode(const std::string & network_interface)
        : Node("cat_control_node"),
          network_interface_(network_interface),  // TODO: Make this a ros param
          low_level_mode_enabler_("/app/sim2real/build/src/release_motion_mode", network_interface_,
              45.0)  // TODO: Move binary into ros package and find it relative to node executable
    {
        static_assert(std::atomic<bool>::is_always_lock_free, "atomic bool is not lock free.");
        init_command_messages();
        load_pytorch_checkpoint();

        RCLCPP_DEBUG(this->get_logger(), "Starting robot state subscriber.");
        robot_state_sub_ = this->create_subscription<unitree_go::msg::LowState>(
            "/lowstate", rclcpp::SensorDataQoS(), std::bind(&CaTControlNode::robot_state_callback, this, std::placeholders::_1));
        RCLCPP_DEBUG(this->get_logger(), "Started robot state subscriber.");

        RCLCPP_DEBUG(this->get_logger(), "Starting robot command publisher.");
        command_publisher = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", rclcpp::SensorDataQoS());
        RCLCPP_DEBUG(this->get_logger(), "Started robot command publisher.");

        std::string error_message;
        RCLCPP_DEBUG(this->get_logger(), "Starting low level control mode enabler process.");
        if (!low_level_mode_enabler_.start(error_message)) {
            fail_node(error_message);
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Started motion switcher helper using interface '%s'.", network_interface_.c_str());

        RCLCPP_DEBUG(this->get_logger(), "Starting robot command publish timer.");
        command_timer_ = this->create_wall_timer(2ms, std::bind(&CaTControlNode::publish_torque_commands, this));
        RCLCPP_DEBUG(this->get_logger(), "Started robot command publish timer.");

        RCLCPP_DEBUG(this->get_logger(), "Starting policy inference / control loop timer.");
        policy_inference_timer_ = this->create_wall_timer(20ms, std::bind(&CaTControlNode::policy_inference_callback, this));
        RCLCPP_DEBUG(this->get_logger(), "Started policy inference / control loop timer.");
        // Important TODO: Add linear interpolation from start pos to standing pos with Kp = 30 and Kd = 1 same way as run_policy.cpp
        // Important TODO: Set default global position targets used by publisher to be current position to avoid sudden movements while inference code
        // is not producign anything
    }

    // TODO: Move to ROS2 params
    std::chrono::microseconds atomic_op_timeout = std::chrono::microseconds{500};
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

private:
    void robot_state_callback(const unitree_go::msg::LowState::SharedPtr msg)
    {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 50, "motor_state[2]: %f", msg->motor_state[2].q);

        // TODO: Remove because this was only used to check if /lowcmd works
        if (low_level_mode_enabled_ && !initial_state_latched_) {
            initial_state_ = *msg;
            start_time_ = this->get_clock()->now().seconds();
            initial_state_latched_ = true;

            RCLCPP_INFO(
                this->get_logger(), "Baseline latched. FR Calf: %f, FL Calf: %f", initial_state_.motor_state[2].q, initial_state_.motor_state[5].q);
        }
    }

    void policy_inference_callback()
    {
    }

    // Sends latest generated actions to the robot at steady 500Hz, as policy only runs at 50Hz.
    void publish_torque_commands()
    {
        if (startup_failed_) { return; }
        if (!low_level_mode_enabled_) {
            std::string error_message;
            const LowLevelModeEnabler::Status status = low_level_mode_enabler_.poll(error_message);

            if (status == LowLevelModeEnabler::Status::Failed) {
                fail_node(error_message);
                return;
            }

            if (status == LowLevelModeEnabler::Status::Running) {
                RCLCPP_INFO_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000, "Waiting for motion switcher helper to release high-level control...");
                return;
            }

            if (status == LowLevelModeEnabler::Status::Succeeded) {
                low_level_mode_enabled_ = true;
                low_level_mode_enabled_time_ = this->get_clock()->now();
                RCLCPP_INFO(this->get_logger(), "Motion switcher helper exited successfully. Starting low-level control.");
            } else {
                return;
            }
        }

        // TODO: Remove because this was only used to check if /lowcmd works
        if (!initial_state_latched_) {
            const double seconds_since_low_level_enabled = (this->get_clock()->now() - low_level_mode_enabled_time_).seconds();

            if (seconds_since_low_level_enabled > initial_state_latch_timeout_seconds_) {
                fail_node("Low-level mode was enabled, but initial /lowstate was not latched in time.");
            }

            return;
        }

        /// TODO: Update based on real policy inference instead of sine wave
        const double t = this->get_clock()->now().seconds() - start_time_;
        const double offset = 0.15 * (1.0 - std::cos(2.0 * M_PI * 0.25 * t));

        const int fr_calf = 2;
        const int fl_calf = 5;

        command_message_.motor_cmd[fr_calf].q = initial_state_.motor_state[fr_calf].q + offset;
        command_message_.motor_cmd[fr_calf].dq = 0.0;
        command_message_.motor_cmd[fr_calf].kp = 30.0;
        command_message_.motor_cmd[fr_calf].kd = 1.0;
        command_message_.motor_cmd[fr_calf].tau = 0.0;

        command_message_.motor_cmd[fl_calf].q = initial_state_.motor_state[fl_calf].q + offset;
        command_message_.motor_cmd[fl_calf].dq = 0.0;
        command_message_.motor_cmd[fl_calf].kp = 30.0;
        command_message_.motor_cmd[fl_calf].kd = 1.0;
        command_message_.motor_cmd[fl_calf].tau = 0.0;

        get_crc(command_message_);
        // Commented out for safety for now
        // command_publisher->publish(command_message_);
    }

    // Init the message struct with appropriate default values
    void init_command_messages()
    {
        command_message_.head[0] = 0xFE;
        command_message_.head[1] = 0xEF;
        command_message_.level_flag = 0xFF;
        command_message_.gpio = 0;

        for (int i = 0; i < 20; i++) {
            command_message_.motor_cmd[i].mode = 0x01;
            command_message_.motor_cmd[i].q = PosStopF;
            command_message_.motor_cmd[i].dq = VelStopF;
            command_message_.motor_cmd[i].kp = 0.0;
            command_message_.motor_cmd[i].kd = 0.0;
            command_message_.motor_cmd[i].tau = 0.0;
        }
    }

    void fail_node(const std::string & message)
    {
        if (startup_failed_) { return; }
        startup_failed_ = true;
        if (command_timer_) { command_timer_->cancel(); }

        low_level_mode_enabler_.stop();
        RCLCPP_ERROR(this->get_logger(), "%s", message.c_str());
        rclcpp::shutdown();
    }

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

        bool checkpoint_in_zero_elevation_map =
            (std::find(checkpoint_filenames_zero_elevation_map.begin(), checkpoint_filenames_zero_elevation_map.end(),
                 checkpoint_path.filename().string()) != checkpoint_filenames_zero_elevation_map.end());
        bool checkpoint_in_proper_elevation_map =
            (std::find(checkpoint_filenames_proper_elevation_map.begin(), checkpoint_filenames_proper_elevation_map.end(),
                 checkpoint_path.filename().string()) != checkpoint_filenames_proper_elevation_map.end());
        if (!checkpoint_in_zero_elevation_map && !checkpoint_in_proper_elevation_map) {
            fail_node(
                "Specified checkpoint file found in neither of the two allowed checkpoint lists, exiting! This is a safety precaution to prevent "
                "passing incorrect observations into a policy, do not circumvent! Simply add the checkpoint to the correct list in the source code "
                "above this message printout.");
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
            fail_node("Failed to load module, exiting.");
        }

        int64_t in_features = -1;
        for (const auto & p : policy_model_.named_parameters(/*recurse=*/true)) {
            if (p.name.ends_with(".weight") && p.value.dim() == 2) {
                in_features = p.value.size(1);
                break;
            }
        }
        if (in_features != observation_dim_no_history && in_features != observation_dim_history) {
            fail_node(std::format("Observation dimension does not match expected value, exiting. in_features={}", in_features));
        }

        int model_observation_dim = in_features;
        RCLCPP_INFO_STREAM(this->get_logger(),
            "Loaded module checkpoint from " << checkpoint_path.string() << "with observation dimension=" << model_observation_dim);
    }


    // TODO: Clean up once motion test is removed in favor of proper policy inference
    const std::string network_interface_;
    const double initial_state_latch_timeout_seconds_ = 2.0;
    LowLevelModeEnabler low_level_mode_enabler_;
    rclcpp::Time low_level_mode_enabled_time_{0, 0, RCL_ROS_TIME};
    bool initial_state_latched_ = false;
    bool low_level_mode_enabled_ = false;
    bool startup_failed_ = false;
    double start_time_ = 0.0;
    unitree_go::msg::LowState initial_state_;

    unitree_go::msg::LowCmd command_message_;
    torch::jit::Module policy_model_;

    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::TimerBase::SharedPtr policy_inference_timer_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr command_publisher;
    // TODO: Add subscriber for elevation map
    // TODO: Add timer for safety checks of last execution time for each pub/sub/timer, as well as safety bounds for state
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