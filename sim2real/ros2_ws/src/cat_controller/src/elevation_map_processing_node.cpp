/*
Author: Loukas Kordos
Disclaimer: This code was proudly written without LLMs :)
*/

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <ament_index_cpp/get_package_prefix.hpp>
#include <atomic>
#include <chrono>
#include <format>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "cat_controller/shutdown_coordinator.hpp"
#include "cat_controller/time_utils.hpp"
#include "cat_perception_msgs/msg/processed_elevation_map.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "grid_map_core/GridMap.hpp"
#include "grid_map_msgs/msg/grid_map.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/exceptions.hpp"
#include "tf2/time.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

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
          robot_base_frame_name_(declare_and_get_param<std::string>("robot_base_frame_name", "Base / CoM frame of the robot", true)),
          processing_frequency_hz_(declare_and_get_param<double>("processing_frequency_hz",
              "How often to process the latest map in Hz. Note that this is independent of how often an elevation map is received, as the "
              "transformation to base will occur more frequently using the latest tf.",
              true)),
          processing_interval_(  // Needs duration double first to avoid truncation
              std::chrono::round<std::chrono::milliseconds>(std::chrono::duration<double, std::milli>(1000.0 / processing_frequency_hz_))),
          tf_lookup_timeout_(
              declare_and_get_param<double>("tf_lookup_timeout", "How long to wait in seconds until shutting down node due to tf timeout", true)),
          processed_map_grid_width_(declare_and_get_param<int>("processed_map_grid_width", "Number of cells in width direction", true)),
          processed_map_grid_height_(declare_and_get_param<int>("processed_map_grid_height", "Number of cells in height direction", true)),
          processed_map_grid_resolution_(
              declare_and_get_param<double>("processed_map_grid_resolution", "Grid resolution (spacing) between cells in meters.", true)),
          elevation_sensor_offset_x_(declare_and_get_param<double>("elevation_sensor_offset_x",
              "In meters. Shifts sensor position relative to base. Policy was trained with offset in positive X direction to focus more data towards "
              "front of robot for walking.",
              true)),
          elevation_sensor_offset_y_(
              declare_and_get_param<double>("elevation_sensor_offset_y", "In meters. Shifts sensor position relative to base", true)),
          invalid_cell_fill_value_(declare_and_get_param<double>(
              "invalid_cell_fill_value", "In meters. Used for Nan/inf in height map, since policy excepts purely numerical data.", true)),
          use_negative_base_height_as_fill_value_(declare_and_get_param<bool>("use_negative_base_height_as_fill_value",
              "If true, invalid_cell_fill_value is NOT used, and invalid cells are instead set to -base_height, which is arguably more accurate",
              true)),
          min_allowed_base_height_(declare_and_get_param<double>("min_allowed_base_height",
              "In meters. Used to safely shut down robot when state estimation / odom reports unreasonable values.", true)),
          max_allowed_base_height_(declare_and_get_param<double>("max_allowed_base_height",
              "In meters. Used to safely shut down robot when state estimation / odom reports unreasonable values.", true)),
          shutdown_coordinator_(
              this->get_logger(), this->get_node_base_interface()->get_context(), [this]() { this->map_processing_timer_->cancel(); })
    {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        double span_x = (processed_map_grid_width_ - 1) * processed_map_grid_resolution_;
        double span_y = (processed_map_grid_height_ - 1) * processed_map_grid_resolution_;
        for (int y_idx = 0; y_idx < processed_map_grid_height_; y_idx++) {
            for (int x_idx = 0; x_idx < processed_map_grid_width_; x_idx++) {
                double x_pos = -span_x / 2.0 + x_idx * processed_map_grid_resolution_ + elevation_sensor_offset_x_;
                double y_pos = -span_y / 2.0 + y_idx * processed_map_grid_resolution_ + elevation_sensor_offset_y_;
                lookup_points_robot_frame_.emplace_back(x_pos, y_pos);
            }
        }

        lookup_points_world_frame_ = lookup_points_robot_frame_;
        processed_elevation_map_values_.resize(processed_map_grid_width_ * processed_map_grid_height_, invalid_cell_fill_value_);

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
        auto processing_start_stamp = this->get_clock()->now();
        if (shutdown_coordinator_.handle_exit_if_requested() || time_utils::shutdown_if_deadline_exceeded(last_processing_callback_time_,
                                                                    std::chrono::milliseconds{2 * processing_interval_}, shutdown_coordinator_))
        {
            return;
        }
        // Use an RCU-like atomic pointer swap. For my own understanding I am writing an explanation here:
        // We bypass locking by allocating a new object on the heap, converting the message without locking yet, and only then atomically store the
        // new pointer globally. That way, the high-frequency timer thread never blocks waiting for a deep copy to finish.
        // I am aware tat atomic shared ptrs are not lock-free, but as the control node will notie latency spikes that might occur due to the heap
        // allocs or waiting for a lock here, this is acceptable.
        std::shared_ptr<grid_map::GridMap> latest_map = global_grid_map_.load();
        // Always need to check for null pointer, e.g. for scenario when no map has been received yet.
        if (!latest_map) {
            RCLCPP_WARN(this->get_logger(), "Global map is null pointer, skipping processing...");
            return;
        }
        // No need for age check of elevation map here since the policy will handle that and stop the robot if the received message is too old

        geometry_msgs::msg::TransformStamped base_to_world_tf;
        try {
            // Arg order is to,from
            base_to_world_tf = tf_buffer_->lookupTransform(
                latest_map->getFrameId(), robot_base_frame_name_, tf2::TimePointZero, tf2::durationFromSec(tf_lookup_timeout_));
        } catch (const tf2::TransformException & e) {
            shutdown_coordinator_.shutdown(std::format("tf lookup failed, exiting. Exception message: {}", e.what()));
            return;
        }
        double fill_value = use_negative_base_height_as_fill_value_ ? -base_to_world_tf.transform.translation.z : invalid_cell_fill_value_;

        if (base_to_world_tf.transform.translation.z < min_allowed_base_height_ ||
            base_to_world_tf.transform.translation.z > max_allowed_base_height_)
        {
            shutdown_coordinator_.shutdown(
                std::format("Reported base height is out of pre-defined safe bounds, exiting for safety. This indicates odom or state estimation is "
                            "having issues or you are climbing a hill :) Reported base z in world frame={}, min_allowed={}, max_allowed={}",
                    base_to_world_tf.transform.translation.z, min_allowed_base_height_, max_allowed_base_height_));
            return;
        }

        double yaw = tf2::getYaw(base_to_world_tf.transform.rotation);
        // ROS2 standard uses right-handed coordinate frame => positive rotation is CCW, so euler convention matches that
        auto rot_base_to_world_yaw = Eigen::Rotation2Dd(yaw);

        // We do not vectorize the rotation math here using Eigen matrices, as the computational cost of ~150 2D rotations is negligible (<1us)
        // and the loop's execution time is mostly dominated by the memory access and interpolation inside grid_map::atPosition().
        for (size_t i = 0; i < lookup_points_world_frame_.size(); i++) {
            lookup_points_world_frame_[i] = rot_base_to_world_yaw * lookup_points_robot_frame_[i];
            lookup_points_world_frame_[i].x() += base_to_world_tf.transform.translation.x;
            lookup_points_world_frame_[i].y() += base_to_world_tf.transform.translation.y;

            grid_map::Position current_pos(lookup_points_world_frame_[i]);
            if (!latest_map->isInside(current_pos)) {
                processed_elevation_map_values_[i] = fill_value;
                continue;  // Skip this position
            }

            double absolute_height = latest_map->atPosition(source_map_layer_name_, current_pos, grid_map::InterpolationMethods::INTER_LINEAR);

            // Check validity of each cell, set to fill value if invalid to avoid interpolation issues.
            // It should almost never happen in practice anyway since the elevation mapping inpainting and min_filter plugins
            // will filter out most and the original map is assumed to be twice as large as the policy region of interest.
            if (!std::isfinite(absolute_height)) {
                processed_elevation_map_values_[i] = fill_value;
                continue;  // Skip this position
            }
            // TODO: Compute is_valid mask for message
            // TODO: Really not sure if I can use the indexing like this, I think this is wrong and needs to be put into the same order as
            // elevation_to_policy
            processed_elevation_map_values_[i] = absolute_height - base_to_world_tf.transform.translation.z;
        }

        cat_perception_msgs::msg::ProcessedElevationMap processed_msg;
        processed_msg.header.frame_id = latest_map->getFrameId();
        processed_msg.header.stamp = processing_start_stamp;
        processed_msg.source_pose_stamp = base_to_world_tf.header.stamp;
        // Pass RCL_ROS_TIME to make it compatible with replaying
        processed_msg.source_map_stamp = rclcpp::Time(static_cast<int64_t>(latest_map->getTimestamp()), RCL_ROS_TIME);
        auto map_size = latest_map->getSize();
        processed_msg.source_size_x = map_size.x();
        processed_msg.source_size_y = map_size.y();
        processed_msg.source_resolution = latest_map->getResolution();
        processed_msg.processed_size_x = processed_map_grid_width_;
        processed_msg.processed_size_y = processed_map_grid_height_;
        processed_msg.processed_resolution = processed_map_grid_resolution_;
        processed_msg.sensor_offset_x = elevation_sensor_offset_x_;
        processed_msg.sensor_offset_y = elevation_sensor_offset_y_;
        processed_msg.fill_value = fill_value;
        processed_msg.seq = processing_iteration_counter_;
        // TODO: Add is_valid mask to the message
        processed_msg.layer_name = source_map_layer_name_;
        processed_msg.data = processed_elevation_map_values_;

        processed_map_publisher_->publish(processed_msg);
        processing_iteration_counter_++;

        // AFTER CONFIRMED WORKING:
        // TODO: profile hotloop using Tracy
        // TODO: Profile atomic shared ptr read and write using tracy
        // TODO: Simplify and cleanup there are differences when running the rosbag-checker on it
    }

    void source_map_subscriber_callback(const grid_map_msgs::msg::GridMap::ConstSharedPtr msg)
    {
        // Use something similar to RCU just with atomic shared pointers as it is perfect for this scenario.
        // For my own understanding, an explanation: Read, i.e. dereference the source pointer Copy, i.e. create a deep copy on the heap of the
        // original message using from_message.
        auto new_source_map = std::make_shared<grid_map::GridMap>();
        grid_map::GridMapRosConverter::fromMessage(*msg, *new_source_map);
        // Check if the layer exists here to avoid having to check in loop, this way all exceptions that can be thrown by grid_map are handled.
        if (!std::ranges::contains(new_source_map->getLayers(), source_map_layer_name_)) {
            shutdown_coordinator_.shutdown(std::format("Desired layer={} not found in elevation map, exiting.", source_map_layer_name_));
            return;
        }
        // Update, i.e. point the current global atomic shared ptr to the newly allocated message on the heap. Since we only read here and then exit,
        // and we only have a single reader, it is guaranteed that after the atomic copy operation, the other reader can do whatever he wants
        global_grid_map_.store(new_source_map);
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
    uint64_t processing_iteration_counter_;

    const std::string source_map_topic_name_;
    const std::string source_map_layer_name_;
    const std::string processed_map_topic_name_;
    const std::string robot_base_frame_name_;
    const double processing_frequency_hz_;
    const std::chrono::milliseconds processing_interval_;  // Needed for wall timer
    const double tf_lookup_timeout_;
    const int processed_map_grid_width_;   // In cells, NOT meters
    const int processed_map_grid_height_;  // In cells, NOT meters
    const double processed_map_grid_resolution_;
    const double elevation_sensor_offset_x_;
    const double elevation_sensor_offset_y_;
    const double invalid_cell_fill_value_;
    const bool use_negative_base_height_as_fill_value_;
    const double min_allowed_base_height_;
    const double max_allowed_base_height_;

    rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr map_subscriber_;
    rclcpp::Publisher<cat_perception_msgs::msg::ProcessedElevationMap>::SharedPtr processed_map_publisher_;
    rclcpp::TimerBase::SharedPtr map_processing_timer_;
    rclcpp::CallbackGroup::SharedPtr map_sub_cbg_;
    rclcpp::CallbackGroup::SharedPtr processing_timer_cbg_;

    // Usually, I would use my own custom timed_atomic here to avoid hanging in a safety critical thread, but since the cat_control_node will have an
    // age check on the received message and simply stop the robot if no new messages arrive, it is acceptable to use a simple atomic shared pointer.
    std::atomic<std::shared_ptr<grid_map::GridMap>> global_grid_map_;
    // List of sample positions in base frame for elevation map. These stay constant in base frame but we need to transform them into world frame
    std::vector<Eigen::Vector2d> lookup_points_robot_frame_;
    // Only to be accessed from timer callback, but elevated to member var to avoid reallocation every iteration
    std::vector<Eigen::Vector2d> lookup_points_world_frame_;
    std::vector<float> processed_elevation_map_values_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

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