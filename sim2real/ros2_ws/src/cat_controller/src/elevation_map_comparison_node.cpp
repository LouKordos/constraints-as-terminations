#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "cat_controller/old_elevation_map_processor.hpp"
#include "cat_controller/shutdown_coordinator.hpp"
#include "cat_controller/stamped_robot_state.hpp"
#include "cat_controller/timed_atomic.hpp"
#include "cat_controller/unitree_msg_utils.hpp"
#include "cat_perception_msgs/msg/processed_elevation_map.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_state.hpp"

using namespace std::chrono_literals;

namespace {

double ros_time_msg_to_seconds(const builtin_interfaces::msg::Time & stamp)
{ return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9; }

double rclcpp_time_to_seconds(const rclcpp::Time & stamp)
{ return static_cast<double>(stamp.nanoseconds()) * 1e-9; }

struct ProcessedMapSample
{
    uint64_t seq{0};
    double header_ts{0.0};
    double source_pose_ts{0.0};
    double source_map_ts{0.0};
    float fill_value{0.0f};
    std::string layer_name;
    std::vector<float> data;
    std::vector<uint8_t> mask;
};

}  // namespace

class ElevationMapComparisonNode : public rclcpp::Node
{
public:
    explicit ElevationMapComparisonNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
        : Node("elevation_map_comparison_node", options),
          processed_map_topic_name_(
              declare_and_get_param<std::string>("processed_map_topic_name", "Processed map topic produced by the new ROS node", true)),
          atomic_op_timeout_us_(declare_and_get_param<int>("atomic_op_timeout_us", "Timeout used for the timed_atomic helpers", true)),
          comparison_frequency_hz_(declare_and_get_param<double>("comparison_frequency_hz", "How often to try matching samples", true)),
          max_match_time_diff_sec_(declare_and_get_param<double>(
              "max_match_time_diff_sec", "Maximum allowed timestamp difference for matching legacy and ROS messages", true)),
          drop_unmatched_after_sec_(
              declare_and_get_param<double>("drop_unmatched_after_sec", "Drop processed samples that stayed unmatched for longer than this", true)),
          max_value_abs_diff_(
              declare_and_get_param<double>("max_value_abs_diff", "Warn if the maximum absolute cell error exceeds this threshold", true)),
          fill_value_epsilon_(declare_and_get_param<double>(
              "fill_value_epsilon", "Two values within this absolute difference from the legacy fill value count as invalid", true)),
          max_buffer_size_(declare_and_get_param<int>("max_buffer_size", "Maximum number of processed samples kept in memory", true)),
          log_all_matches_(declare_and_get_param<bool>("log_all_matches", "If true, print an INFO line for every successful match", true)),
          comparison_timer_period_(
              std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / comparison_frequency_hz_))),
          atomic_op_timeout_(std::chrono::microseconds(atomic_op_timeout_us_)),
          legacy_grid_sink_(std::vector<float>(legacy_grid_total_size_, -0.27f)),
          shutdown_coordinator_(this->get_logger(), this->get_node_base_interface()->get_context(), [this]() {
              if (comparison_timer_) { comparison_timer_->cancel(); }
          })
    {
        if (comparison_frequency_hz_ <= 0.0) { throw std::runtime_error("comparison_frequency_hz must be > 0"); }
        if (max_match_time_diff_sec_ < 0.0) { throw std::runtime_error("max_match_time_diff_sec must be >= 0"); }
        if (drop_unmatched_after_sec_ < 0.0) { throw std::runtime_error("drop_unmatched_after_sec must be >= 0"); }
        if (max_buffer_size_ <= 0) { throw std::runtime_error("max_buffer_size must be > 0"); }

        const char * ros_log_dir = std::getenv("ROS_LOG_DIR");
        std::string log_directory = ros_log_dir ? std::string(ros_log_dir) + "/" : "/tmp/";

        legacy_processor_ = std::make_unique<ElevationMapProcessor>(
            log_directory, "min_filter", 1, shutdown_coordinator_, this->get_logger(), global_robot_state_, legacy_grid_sink_);

        lowstate_subscriber_ = this->create_subscription<unitree_go::msg::LowState>(
            "/lowstate", rclcpp::SensorDataQoS(), std::bind(&ElevationMapComparisonNode::lowstate_callback, this, std::placeholders::_1));

        processed_map_subscriber_ = this->create_subscription<cat_perception_msgs::msg::ProcessedElevationMap>(processed_map_topic_name_,
            rclcpp::SensorDataQoS().keep_last(50), std::bind(&ElevationMapComparisonNode::processed_map_callback, this, std::placeholders::_1));

        comparison_timer_ =
            this->create_wall_timer(comparison_timer_period_, std::bind(&ElevationMapComparisonNode::comparison_timer_callback, this));

        std::string param_dump = "=== Node Parameters ===\n";
        auto param_list = this->list_parameters(std::vector<std::string>{}, 10);
        for (const auto & param_name : param_list.names) {
            rclcpp::Parameter param;
            if (this->get_parameter(param_name, param)) { param_dump += fmt::format("  {}: {}\n", param_name, param.value_to_string()); }
        }
        param_dump += "============================";
        RCLCPP_INFO(this->get_logger(), "\n%s", param_dump.c_str());
        RCLCPP_WARN(this->get_logger(),
            "This node is only meant for temporary on-robot comparison. For apples-to-apples results, configure the new processed map node to use "
            "the same fill-value convention as the legacy Python pipeline (normally fixed -0.27 for the relative map).");
    }

private:
    template <typename T>
    T declare_and_get_param(const std::string & name, const std::string & description = "", bool read_only = false)
    {
        rcl_interfaces::msg::ParameterDescriptor desc;
        desc.description = description;
        desc.read_only = read_only;

        try {
            this->declare_parameter<T>(name, desc);
            return this->get_parameter(name).get_value<T>();
        } catch (const rclcpp::exceptions::ParameterUninitializedException &) {
            RCLCPP_FATAL(this->get_logger(), "CRITICAL: Mandatory parameter '%s' is missing from config!", name.c_str());
            throw;
        } catch (const rclcpp::exceptions::InvalidParameterTypeException &) {
            RCLCPP_FATAL(this->get_logger(), "CRITICAL: Parameter '%s' has the wrong type in YAML!", name.c_str());
            throw;
        }
    }

    void lowstate_callback(const unitree_go::msg::LowState::SharedPtr msg)
    {
        auto steady_now = std::chrono::steady_clock::now();
        auto stamped_state = stamped_state_from_lowstate(*msg, lowstate_counter_++, steady_now);
        global_robot_state_.try_store_for(stamped_state, atomic_op_timeout_);
    }

    void processed_map_callback(const cat_perception_msgs::msg::ProcessedElevationMap::SharedPtr msg)
    {
        ProcessedMapSample sample;
        sample.seq = msg->seq;
        sample.header_ts = ros_time_msg_to_seconds(msg->header.stamp);
        sample.source_pose_ts = ros_time_msg_to_seconds(msg->source_pose_stamp);
        sample.source_map_ts = rclcpp_time_to_seconds(rclcpp::Time(msg->source_map_stamp, RCL_ROS_TIME));
        sample.fill_value = msg->fill_value;
        sample.layer_name = msg->layer_name;
        sample.data = msg->data;
        sample.mask = msg->is_valid_mask;

        std::lock_guard<std::mutex> lock(processed_samples_mutex_);
        processed_samples_.push_back(std::move(sample));
        while (processed_samples_.size() > static_cast<size_t>(max_buffer_size_)) { processed_samples_.pop_front(); }
    }

    void comparison_timer_callback()
    {
        if (shutdown_coordinator_.handle_exit_if_requested()) { return; }

        const std::vector<LegacyElevationSample> legacy_samples = legacy_processor_->get_recent_samples_copy();
        if (legacy_samples.empty()) { return; }

        std::lock_guard<std::mutex> lock(processed_samples_mutex_);
        while (!processed_samples_.empty()) {
            const ProcessedMapSample & processed_sample = processed_samples_.front();

            size_t best_index = 0;
            double best_time_diff = std::numeric_limits<double>::infinity();
            bool found_match = false;

            for (size_t i = 0; i < legacy_samples.size(); i++) {
                const double current_diff = std::abs(processed_sample.source_map_ts - legacy_samples[i].map_ts);
                if (current_diff < best_time_diff) {
                    best_time_diff = current_diff;
                    best_index = i;
                    found_match = true;
                }
            }

            if (!found_match) { break; }

            if (best_time_diff <= max_match_time_diff_sec_) {
                compare_and_log(processed_sample, legacy_samples[best_index], best_time_diff);
                processed_samples_.pop_front();
                continue;
            }

            const double latest_legacy_map_ts = legacy_samples.back().map_ts;
            if ((latest_legacy_map_ts - processed_sample.source_map_ts) > drop_unmatched_after_sec_) {
                dropped_samples_++;
                RCLCPP_WARN(this->get_logger(),
                    "Dropping unmatched processed sample seq=%llu. source_map_ts=%.9f, best legacy diff=%.6fs, dropped=%llu",
                    static_cast<unsigned long long>(processed_sample.seq), processed_sample.source_map_ts, best_time_diff,
                    static_cast<unsigned long long>(dropped_samples_));
                processed_samples_.pop_front();
                continue;
            }

            break;
        }
    }

    void compare_and_log(const ProcessedMapSample & processed_sample, const LegacyElevationSample & legacy_sample, double matched_time_diff)
    {
        compared_samples_++;

        if (processed_sample.data.size() != legacy_sample.grid.size()) {
            failed_samples_++;
            RCLCPP_ERROR(this->get_logger(), "Size mismatch for seq=%llu: processed size=%zu, legacy size=%zu, map_ts_diff=%.6fs",
                static_cast<unsigned long long>(processed_sample.seq), processed_sample.data.size(), legacy_sample.grid.size(), matched_time_diff);
            return;
        }

        if (!processed_sample.mask.empty() && processed_sample.mask.size() != processed_sample.data.size()) {
            failed_samples_++;
            RCLCPP_ERROR(this->get_logger(), "Processed message seq=%llu has invalid mask size. data=%zu, mask=%zu",
                static_cast<unsigned long long>(processed_sample.seq), processed_sample.data.size(), processed_sample.mask.size());
            return;
        }

        size_t valid_mask_mismatch_count = 0;
        size_t large_diff_count = 0;
        size_t both_valid_count = 0;
        double max_abs_diff_all_cells = 0.0;
        double max_abs_diff_both_valid = 0.0;
        double mean_abs_diff_all_cells = 0.0;
        double mean_abs_diff_both_valid = 0.0;
        size_t max_abs_diff_index = 0;

        for (size_t i = 0; i < processed_sample.data.size(); i++) {
            const float processed_value = processed_sample.data[i];
            const float legacy_value = legacy_sample.grid[i];

            const bool processed_valid = processed_sample.mask.empty() ? std::isfinite(processed_value) : (processed_sample.mask[i] != 0u);
            const bool legacy_valid = std::isfinite(legacy_value) && (std::abs(legacy_value - legacy_sample.fill_value) > fill_value_epsilon_);

            if (processed_valid != legacy_valid) { valid_mask_mismatch_count++; }

            const double abs_diff = std::abs(static_cast<double>(processed_value) - static_cast<double>(legacy_value));
            mean_abs_diff_all_cells += abs_diff;
            if (abs_diff > max_abs_diff_all_cells) {
                max_abs_diff_all_cells = abs_diff;
                max_abs_diff_index = i;
            }
            if (abs_diff > max_value_abs_diff_) { large_diff_count++; }

            if (processed_valid && legacy_valid) {
                both_valid_count++;
                mean_abs_diff_both_valid += abs_diff;
                if (abs_diff > max_abs_diff_both_valid) { max_abs_diff_both_valid = abs_diff; }
            }
        }

        mean_abs_diff_all_cells /= static_cast<double>(processed_sample.data.size());
        if (both_valid_count > 0) {
            mean_abs_diff_both_valid /= static_cast<double>(both_valid_count);
        } else {
            mean_abs_diff_both_valid = 0.0;
        }

        const bool pass = (valid_mask_mismatch_count == 0) && (max_abs_diff_all_cells <= max_value_abs_diff_);

        if (!pass) {
            failed_samples_++;
            RCLCPP_ERROR(this->get_logger(),
                "COMPARE FAIL seq=%llu layer=%s map_ts_new=%.9f map_ts_old=%.9f diff=%.6fs max_abs_all=%.8f mean_abs_all=%.8f "
                "max_abs_valid=%.8f mean_abs_valid=%.8f valid_mask_mismatch=%zu large_diff_count=%zu max_diff_index=%zu "
                "processed_fill=%.5f legacy_fill=%.5f failed=%llu compared=%llu",
                static_cast<unsigned long long>(processed_sample.seq), processed_sample.layer_name.c_str(), processed_sample.source_map_ts,
                legacy_sample.map_ts, matched_time_diff, max_abs_diff_all_cells, mean_abs_diff_all_cells, max_abs_diff_both_valid,
                mean_abs_diff_both_valid, valid_mask_mismatch_count, large_diff_count, max_abs_diff_index, processed_sample.fill_value,
                legacy_sample.fill_value, static_cast<unsigned long long>(failed_samples_), static_cast<unsigned long long>(compared_samples_));
            return;
        }

        if (log_all_matches_) {
            RCLCPP_INFO(this->get_logger(),
                "COMPARE OK seq=%llu layer=%s map_ts_diff=%.6fs max_abs_all=%.8f mean_abs_all=%.8f max_abs_valid=%.8f mean_abs_valid=%.8f",
                static_cast<unsigned long long>(processed_sample.seq), processed_sample.layer_name.c_str(), matched_time_diff, max_abs_diff_all_cells,
                mean_abs_diff_all_cells, max_abs_diff_both_valid, mean_abs_diff_both_valid);
        } else if ((compared_samples_ % 50u) == 0u) {
            RCLCPP_INFO(this->get_logger(), "Compared %llu samples so far, failures=%llu, dropped=%llu",
                static_cast<unsigned long long>(compared_samples_), static_cast<unsigned long long>(failed_samples_),
                static_cast<unsigned long long>(dropped_samples_));
        }
    }

    static constexpr int legacy_grid_width_ = 13;
    static constexpr int legacy_grid_height_ = 11;
    static constexpr int legacy_grid_total_size_ = legacy_grid_width_ * legacy_grid_height_;

    const std::string processed_map_topic_name_;
    const int atomic_op_timeout_us_;
    const double comparison_frequency_hz_;
    const double max_match_time_diff_sec_;
    const double drop_unmatched_after_sec_;
    const double max_value_abs_diff_;
    const double fill_value_epsilon_;
    const int max_buffer_size_;
    const bool log_all_matches_;

    const std::chrono::nanoseconds comparison_timer_period_;
    const std::chrono::microseconds atomic_op_timeout_;

    uint64_t lowstate_counter_{0};
    uint64_t compared_samples_{0};
    uint64_t failed_samples_{0};
    uint64_t dropped_samples_{0};

    timed_atomic<stamped_robot_state> global_robot_state_{};
    timed_atomic<std::vector<float>> legacy_grid_sink_;

    std::unique_ptr<ElevationMapProcessor> legacy_processor_;

    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_subscriber_;
    rclcpp::Subscription<cat_perception_msgs::msg::ProcessedElevationMap>::SharedPtr processed_map_subscriber_;
    rclcpp::TimerBase::SharedPtr comparison_timer_;

    std::mutex processed_samples_mutex_;
    std::deque<ProcessedMapSample> processed_samples_;

    ShutdownCoordinator shutdown_coordinator_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ElevationMapComparisonNode>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}