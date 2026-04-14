#include <chrono>
#include <cmath>
#include <thread>
#include <memory>
#include <string>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "cat_controller/motor_crc.h" // Copied from go2 repo because its needed for sending valid motor commands and they do not install these header files automatically

using namespace std::chrono_literals;

class CaTControlNode : public rclcpp::Node
{
public:
    CaTControlNode() : Node("cat_control_node")
    {
        init_command();
        rclcpp::SensorDataQoS best_effort_qos{}; // keep-last, depth 5, best effort, volatile
        robot_state_sub_ = this->create_subscription<unitree_go::msg::LowState>("/lowstate", best_effort_qos, std::bind(&CaTControlNode::robot_state_callback, this, std::placeholders::_1));
        command_timer_ = this->create_wall_timer(2ms, std::bind(&CaTControlNode::publish_torque_commands, this));
        command_publisher = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", best_effort_qos);
    }

private:
    void robot_state_callback(const unitree_go::msg::LowState::SharedPtr msg)
    {
        unitree_go::msg::LowState state = *msg;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, "RPY[0]: %f", state.imu_state.rpy[0]);

        if (!initial_state_latched_)
        {
            initial_state_ = *msg;
            start_time_ = this->get_clock()->now().seconds();
            initial_state_latched_ = true;

            RCLCPP_INFO(this->get_logger(), "Baseline latched. FR Calf: %f, FL Calf: %f",
                        initial_state_.motor_state[2].q, initial_state_.motor_state[5].q);
        }
    }
    void publish_torque_commands()
    {
        RCLCPP_INFO(this->get_logger(), "Would publish now at time=%f", this->get_clock()->now().seconds());
        if (!initial_state_latched_)
        {
            return;
        }

        // Calculate elapsed time since we latched the state
        double t = this->get_clock()->now().seconds() - start_time_;

        // 3. Create a smooth wave that goes from 0.0 to +0.3 radians and back every 4 seconds.
        // Positive calf angle delta pushes the foot down, lifting the chassis.
        double offset = 0.15 * (1.0 - std::cos(2.0 * M_PI * 0.25 * t));

        int fr_calf = 2;
        int fl_calf = 5;

        command_message_.motor_cmd[fr_calf].q = initial_state_.motor_state[fr_calf].q + offset;
        command_message_.motor_cmd[fr_calf].dq = 0.0;
        command_message_.motor_cmd[fr_calf].kp = 25.0;
        command_message_.motor_cmd[fr_calf].kd = 1.0;
        command_message_.motor_cmd[fr_calf].tau = 0.0;

        command_message_.motor_cmd[fl_calf].q = initial_state_.motor_state[fl_calf].q + offset;
        command_message_.motor_cmd[fl_calf].dq = 0.0;
        command_message_.motor_cmd[fl_calf].kp = 25.0;

        command_message_.motor_cmd[fl_calf].kd = 1.0;
        command_message_.motor_cmd[fl_calf].tau = 0.0;

        get_crc(command_message_);
        command_publisher->publish(command_message_);
    }
    void init_command()
    {
        command_message_.head[0] = 0xFE;
        command_message_.head[1] = 0xEF;
        command_message_.level_flag = 0xFF;
        command_message_.gpio = 0;

        for (int i = 0; i < 20; i++)
        {
            command_message_.motor_cmd[i].mode = 0x01;

            command_message_.motor_cmd[i].q = PosStopF;
            command_message_.motor_cmd[i].dq = VelStopF;
            command_message_.motor_cmd[i].kp = 0.0;
            command_message_.motor_cmd[i].kd = 0.0;
            command_message_.motor_cmd[i].tau = 0.0;
        }
    }

    unitree_go::msg::LowCmd command_message_;
    bool initial_state_latched_ = false;
    double start_time_ = 0.0;
    unitree_go::msg::LowState initial_state_;

    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr command_publisher;
    // TODO: Add subscriber for elevation map
    // TODO: Timer for inference
    // TODO: Timer for safety checks of last execution time for each pub/sub/timer, as well as safety bounds for state
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CaTControlNode>());
    rclcpp::shutdown();
    return 0;
}