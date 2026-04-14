#include <chrono>
#include <cmath>
#include <thread>
#include <memory>
#include <string>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

// #include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/low_cmd.hpp"
// #include "cat_controller/motor_crc.h" // Copied from go2 repo because its needed for sending valid motor commands and they do not install these header files automatically

using namespace std::chrono_literals;

class CaTControlNode : public rclcpp::Node {
    public:
        CaTControlNode() : Node("cat_control_node") {
            rclcpp::QoS best_effort_qos(10); // Queue size 10;
            robot_state_sub_ = this->create_subscription<unitree_go::msg::LowState>("/lowstate", best_effort_qos, std::bind(&CaTControlNode::robot_state_callback, this, std::placeholders::_1));
        }
    private:
        void robot_state_callback(const unitree_go::msg::LowState::SharedPtr msg) {
            unitree_go::msg::LowState state = *msg;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, "RPY[0]: %f", state.imu_state.rpy[0]);
        }

        rclcpp::TimerBase::SharedPtr command_timer_;
        rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CaTControlNode>());
    rclcpp::shutdown();
    return 0;
}