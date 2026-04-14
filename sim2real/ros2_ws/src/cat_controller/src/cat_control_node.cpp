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
            robot_state_timer_ = this->create_wall_timer(2000ms, std::bind(&CaTControlNode::read_robot_state, this));
        }
    private:
        void read_robot_state() {
            RCLCPP_INFO(this->get_logger(), "Test");
        }

        rclcpp::TimerBase::SharedPtr robot_state_timer_;
        rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr robot_state_sub_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CaTControlNode>());
    rclcpp::shutdown();
    return 0;
}