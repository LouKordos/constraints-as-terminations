#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
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

        rclcpp::SensorDataQoS best_effort_qos{};
        RCLCPP_DEBUG(this->get_logger(), "Starting robot state subscriber.");
        robot_state_sub_ = this->create_subscription<unitree_go::msg::LowState>(
            "/lowstate", best_effort_qos, std::bind(&CaTControlNode::robot_state_callback, this, std::placeholders::_1));
        RCLCPP_DEBUG(this->get_logger(), "Started robot state subscriber.");
        RCLCPP_DEBUG(this->get_logger(), "Starting robot command publisher.");
        command_publisher = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", best_effort_qos);
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
    }

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
        // TODO: Wait until low level control mode is enabled same way as publish_torque_commands
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

    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::TimerBase::SharedPtr policy_inference_timer_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr command_publisher;
    // TODO: Add subscriber for elevation map
    // TODO: Timer for safety checks of last execution time for each pub/sub/timer, as well as safety bounds for state
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