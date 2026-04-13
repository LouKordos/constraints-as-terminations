#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class CaTControlNode : public rclcpp::Node {
    public:
        CaTControlNode() : Node("cat_control_node") {
            auto timer_callback = [this]() -> void {
                RCLCPP_INFO(this->get_logger(), "Hello world");
            };
            timer_ = this->create_wall_timer(500ms, timer_callback);
        }
    private:
        rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CaTControlNode>());
    rclcpp::shutdown();
    return 0;
}