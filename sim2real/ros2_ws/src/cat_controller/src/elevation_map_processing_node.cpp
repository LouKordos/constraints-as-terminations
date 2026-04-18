#include <ament_index_cpp/get_package_prefix.hpp>

#include "rclcpp/rclcpp.hpp"

class ElevationMapProcessingNode : public rclcpp::Node
{
public:
    explicit ElevationMapProcessingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions()) : Node("elevation_map_processing_node", options)
    {
    }

private:
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ElevationMapProcessingNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}