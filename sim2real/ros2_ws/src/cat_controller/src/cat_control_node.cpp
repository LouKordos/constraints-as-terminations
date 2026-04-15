#include <limits.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <functional>
#include <memory>
#include <string>

#include "cat_controller/motor_crc.h"  // Copied from go2 repo because its needed for sending valid motor commands and they do not install these header files automatically
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

using namespace std::chrono_literals;

class CaTControlNode : public rclcpp::Node
{
public:
    explicit CaTControlNode(const std::string & network_interface)
        : Node("cat_control_node"), network_interface_(network_interface)  // TODO: Make this a ros param
    {
        init_command();

        rclcpp::SensorDataQoS best_effort_qos{};
        robot_state_sub_ = this->create_subscription<unitree_go::msg::LowState>(
            "/lowstate", best_effort_qos, std::bind(&CaTControlNode::robot_state_callback, this, std::placeholders::_1));

        // Required to switch from high level "sport mode" state to low level control so that policy actions are applied
        start_motion_switcher_process();
        // Only start after blocking motion switcher has succeded. ANYTHING control related should only start after this!
        command_timer_ = this->create_wall_timer(2ms, std::bind(&CaTControlNode::publish_torque_commands, this));
        command_publisher = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", best_effort_qos);
    }

    ~CaTControlNode() override { stop_motion_switcher_process(SIGKILL); }

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

    // Sends latest generated actions to the robot at steady 500Hz, as policy only runs at 50Hz.
    void publish_torque_commands()
    {
        if (startup_failed_) { return; }
        if (!low_level_mode_enabled_) {
            monitor_motion_switcher_process();
            return;
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
    void init_command()
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

    /*
    Low-level control is enabled by an external motion-switch helper on purpose. I originally tried to disable `sport_mode` from the ROS side, but
    although the API path responded, the robot never actually handed over control. The working mechanism was Unitree's SDK2
    MotionSwitcherClient::ReleaseMode(), but calling that inside the ROS process was not robust because the ROS environment pulled in different DDS
    libraries than the standalone SDK2 setup, leading to version-mismatch crashes.

    This is why a separate `release_motion_mode` helper process is started through the dynamic loader with an explicit library path so it uses the
    known-good SDK2 DDS libraries. The node then waits for that helper to exit successfully before sending any `/lowcmd` messages.

    This node must also not remain in a stale half-started state, so if the helper does not succeed within a bounded time, the node fails and shuts
    down instead of silently sitting around without valid control authority.
    */
    void start_motion_switcher_process()
    {
        const std::string helper_binary_path = "/app/sim2real/build/src/release_motion_mode";
        const pid_t child_pid = fork();
        if (child_pid < 0) {
            fail_node(std::string("fork() failed while starting motion switcher helper: ") + std::strerror(errno));
            return;
        }

        if (child_pid == 0) {
            // Overriding dynamic lib is required because of a version mismatch between cyclonedds used by ROS2 vs. the one used by Unitree.
            // The binary then segfaults because one of the cyclonedds so libs comes from the global install and the other one from ROS.
            const char * loader_path = "/lib64/ld-linux-x86-64.so.2";
            const char * library_path = "/usr/local/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu";
            execl(loader_path, loader_path, "--library-path", library_path, helper_binary_path.c_str(), network_interface_.c_str(),
                static_cast<char *>(nullptr));
            std::cerr << "execl() failed for release_motion_mode via ld-linux: " << std::strerror(errno) << std::endl;
            _exit(127);
        }

        motion_switcher_pid_ = child_pid;
        motion_switcher_start_time_ = this->get_clock()->now();
        RCLCPP_INFO(this->get_logger(), "Started motion switcher helper with pid=%d using interface '%s'.", static_cast<int>(motion_switcher_pid_),
            network_interface_.c_str());
    }

    void monitor_motion_switcher_process()
    {
        if (motion_switcher_pid_ <= 0) {
            fail_node("Motion switcher helper process was never started correctly.");
            return;
        }

        int status = 0;
        const pid_t wait_result = waitpid(motion_switcher_pid_, &status, WNOHANG);
        if (wait_result == 0) {
            const double elapsed_seconds = (this->get_clock()->now() - motion_switcher_start_time_).seconds();
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for motion switcher helper to release high-level control...");
            if (elapsed_seconds > motion_switcher_timeout_seconds_) {
                stop_motion_switcher_process(SIGKILL);
                fail_node("Timed out waiting for motion switcher helper to enable low-level control mode.");
            }

            return;
        }

        if (wait_result < 0) {
            fail_node(std::string("waitpid() failed while monitoring motion switcher helper: ") + std::strerror(errno));
            return;
        }

        motion_switcher_pid_ = -1;
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            low_level_mode_enabled_ = true;
            low_level_mode_enabled_time_ = this->get_clock()->now();
            RCLCPP_INFO(this->get_logger(), "Motion switcher helper exited successfully. Starting low-level control.");
            return;
        }
        if (WIFEXITED(status)) {
            fail_node("Motion switcher helper exited with failure code " + std::to_string(WEXITSTATUS(status)) +
                      ". Low-level control mode was not enabled.");
            return;
        }

        if (WIFSIGNALED(status)) {
            fail_node(
                "Motion switcher helper was terminated by signal " + std::to_string(WTERMSIG(status)) + ". Low-level control mode was not enabled.");
            return;
        }

        fail_node("Motion switcher helper ended in an unknown state.");
    }

    void stop_motion_switcher_process(int signal_number)
    {
        if (motion_switcher_pid_ <= 0) { return; }
        kill(motion_switcher_pid_, signal_number);
        int status = 0;
        waitpid(motion_switcher_pid_, &status, 0);
        motion_switcher_pid_ = -1;
    }

    void fail_node(const std::string & message)
    {
        if (startup_failed_) { return; }
        startup_failed_ = true;
        if (command_timer_) { command_timer_->cancel(); }

        stop_motion_switcher_process(SIGKILL);
        RCLCPP_ERROR(this->get_logger(), "%s", message.c_str());
        rclcpp::shutdown();
    }

    const std::string network_interface_;
    const double motion_switcher_timeout_seconds_ = 45.0;
    const double initial_state_latch_timeout_seconds_ = 2.0;

    unitree_go::msg::LowCmd command_message_;
    bool initial_state_latched_ = false;
    bool low_level_mode_enabled_ = false;
    bool startup_failed_ = false;
    double start_time_ = 0.0;
    unitree_go::msg::LowState initial_state_;

    pid_t motion_switcher_pid_ = -1;
    rclcpp::Time motion_switcher_start_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time low_level_mode_enabled_time_{0, 0, RCL_ROS_TIME};

    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr command_publisher;
    // TODO: Add subscriber for elevation map
    // TODO: Timer for inference
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