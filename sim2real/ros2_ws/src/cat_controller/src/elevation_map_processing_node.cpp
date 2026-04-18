#include <ament_index_cpp/get_package_prefix.hpp>
#include <chrono>

#include "cat_perception_msgs/msg/processed_elevation_map.hpp"
#include "grid_map_msgs/msg/grid_map.hpp"
#include "rclcpp/rclcpp.hpp"

class ElevationMapProcessingNode : public rclcpp::Node
{
public:
    explicit ElevationMapProcessingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions()) : Node("elevation_map_processing_node", options)
    {
        // TODO: Init elevation map processor
        // TODO: setup sub,pub,timer
        // TODO: Initialize and RESERVE working copy processed elevation map vector to avoid heap allocs
    }

private:
    void publish_processed_map()
    {
        // TODO: Load global atomic map, do NOT copy just use const &
        // TODO: Probably keep private working copy of current_procesed_map and call reserve in constructor on that to avoid heap allocs? Downside is
        // thread safety but since node is self contained it should be fine
        // TODO: Call map processor to generate processed map
        // TODO: Create custom msg, publish, log
    }

    // How can I improve naming here?
    void map_subscriber_callback(const cat_perception_msgs::msg::ProcessedElevationMap msg)
    {
        // TODO: Check callback age using helper!
        // TODO: Convert elevation map to grid_map type
        // TODO: store globally using atomic shared ptr probably, need to check best way here
        // TODO: Log info
    }

    rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr map_subscriber_;
    rclcpp::Publisher<cat_perception_msgs::msg::ProcessedElevationMap>::SharedPtr processed_map_publisher_;
    rclcpp::TimerBase::SharedPtr map_processing_timer_;
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