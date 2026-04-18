#include <fmt/core.h>
#include <fmt/ranges.h>

#include <ament_index_cpp/get_package_prefix.hpp>
#include <chrono>
#include <string>

#include "cat_controller/shutdown_coordinator.hpp"
#include "cat_controller/time_utils.hpp"
#include "cat_perception_msgs/msg/processed_elevation_map.hpp"
#include "grid_map_msgs/msg/grid_map.hpp"
#include "rclcpp/rclcpp.hpp"

class ElevationMapProcessingNode : public rclcpp::Node
{
public:
    explicit ElevationMapProcessingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
        : Node("elevation_map_processing_node", options),
          source_map_topic_name_(declare_and_get_param<std::string>("source_map_topic_name", "Topic name of source elevation map", true)),
          source_map_layer_name_(declare_and_get_param<std::string>(
              "source_map_layer_name", "Layer name of source elevation map such as min_filter, smooth, etc.", true)),
          processed_map_topic_name_(
              declare_and_get_param<std::string>("processed_map_topic_name", "Where to publish the processed elevation map messages", true)),
          processing_frequency_hz_(declare_and_get_param<double>("processing_frequency_hz",
              "How often to process the latest map in Hz. Note that this is independent of how often an elevation map is received, as the "
              "transformation to body will occur more frequently using the latest tf.",
              true)),
          processing_interval_(  // Needs duration double first to avoid truncation
              std::chrono::round<std::chrono::milliseconds>(std::chrono::duration<double, std::milli>(1000.0 / processing_frequency_hz_))),
          processed_map_grid_width_(declare_and_get_param<int>("processed_map_grid_width", "Number of cells in width direction", true)),
          processed_map_grid_height_(declare_and_get_param<int>("processed_map_grid_height", "Number of cells in height direction", true)),
          processed_grid_resolution_(
              declare_and_get_param<double>("processed_grid_resolution", "Grid resolution (spacing) between cells in meters.", true)),
          elevation_sensor_offset_x_(declare_and_get_param<double>("elevation_sensor_offset_x",
              "In meters. Shifts sensor position relative to base. Policy was trained with offset in positive X direction to focus more data towards "
              "front of robot for walking.",
              true)),
          elevation_sensor_offset_y_(
              declare_and_get_param<double>("elevation_sensor_offset_y", "In meters. Shifts sensor position relative to base", true)),
          elevation_sensor_offset_z_(
              declare_and_get_param<double>("elevation_sensor_offset_z", "In meters. Shifts sensor position relative to base", true)),
          invalid_cell_fill_value_(declare_and_get_param<double>(
              "invalid_cell_fill_value", "In meters. Used for Nan/inf in height map, since policy excepts purely numerical data.", true)),
          shutdown_coordinator_(
              this->get_logger(), this->get_node_base_interface()->get_context(), [this]() { this->map_processing_timer_->cancel(); })
    {
        RCLCPP_DEBUG(this->get_logger(), "Starting elevation map subscriber.");
        this->map_sub_cbg_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions map_sub_options;
        map_sub_options.callback_group = map_sub_cbg_;
        map_subscriber_ = this->create_subscription<grid_map_msgs::msg::GridMap>(source_map_topic_name_, rclcpp::SensorDataQoS().keep_last(1),
            std::bind(&ElevationMapProcessingNode::source_map_subscriber_callback, this, std::placeholders::_1), map_sub_options);
        RCLCPP_DEBUG(this->get_logger(), "Successfully started elevation map subscriber.");

        RCLCPP_DEBUG(this->get_logger(), "Starting processed elevation map publisher.");
        processed_map_publisher_ =
            this->create_publisher<cat_perception_msgs::msg::ProcessedElevationMap>(processed_map_topic_name_, rclcpp::SensorDataQoS().keep_last(1));
        RCLCPP_DEBUG(this->get_logger(), "Successfully started processed elevation map publisher.");

        RCLCPP_DEBUG(this->get_logger(), "Starting processing timer.");
        this->processing_timer_cbg_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        map_processing_timer_ = this->create_wall_timer(
            processing_interval_, std::bind(&ElevationMapProcessingNode::process_and_publish_map, this), processing_timer_cbg_);
        RCLCPP_DEBUG(this->get_logger(), "Successfully started processing timer.");

        // TODO: Initialize and RESERVE working copy processed elevation map vector to avoid heap allocs

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
    // Takes a GridMap and applies required transformation and interpolation to prepare it for policy inference
    void process_and_publish_map()
    {
        auto steady_now = std::chrono::steady_clock::now();
        if (shutdown_coordinator_.handle_exit_if_requested() || time_utils::shutdown_if_deadline_exceeded(last_processing_callback_time_,
                                                                    std::chrono::milliseconds{2 * processing_interval_}, shutdown_coordinator_))
        {
            return;
        }
        // TODO: Load global atomic map, copy into working copy to avoid
        // No need for age check of global elevation map here, since the policy will handle that and stop the robot if the received message is too old

        // TODO: Probably keep private working copy of current_procesed_map and call reserve in constructor on that to avoid heap allocs?
        // TODO: Process map to convert into polciy observation format as in python

        // TODO: for interpolation, first implement a manual bilinear interpolation that does the same as python
        // Keep this TODO: Test with grid_map builtin interpolation and check the differences because that is cleaner. I don't mind using nearest
        // neighbor since invalid cells are very rare in my use case anyway

        // TODO: gridmap heavily relies on exceptions, make sure to fully catch all of them, log and return only std::expected, NODE SHOULD NOT THROW
        // EXCEPTIONS BUT HANDLE USING shutdown_coordinator instead! Proboably use IsInside to avoid try catch block around hot loop of ~150 lookups?

        // TODO: After confirmed working, start simplifying and cleaning up until there are differences when running the rosbag-checker on it
    }

    void source_map_subscriber_callback(const grid_map_msgs::msg::GridMap::ConstSharedPtr msg)
    {
        // TODO: store globally using atomic shared ptr probably, need to check best approach here to avoid expensive copies and heap allocs here.
        // TODO: Log info
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

    std::chrono::steady_clock::time_point last_processing_callback_time_;

    const std::string source_map_topic_name_;
    const std::string source_map_layer_name_;
    const std::string processed_map_topic_name_;
    const double processing_frequency_hz_;
    const std::chrono::milliseconds processing_interval_;  // Needed for wall timer
    const int processed_map_grid_width_;
    const int processed_map_grid_height_;
    const double processed_grid_resolution_;
    const double elevation_sensor_offset_x_;
    const double elevation_sensor_offset_y_;
    const double elevation_sensor_offset_z_;
    const double invalid_cell_fill_value_;

    rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr map_subscriber_;
    rclcpp::Publisher<cat_perception_msgs::msg::ProcessedElevationMap>::SharedPtr processed_map_publisher_;
    rclcpp::TimerBase::SharedPtr map_processing_timer_;
    rclcpp::CallbackGroup::SharedPtr map_sub_cbg_;
    rclcpp::CallbackGroup::SharedPtr processing_timer_cbg_;

    ShutdownCoordinator shutdown_coordinator_;
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