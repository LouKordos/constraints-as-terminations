#pragma once
#include <chrono>
#include <rclcpp/rclcpp.hpp>

#include "cat_controller/shutdown_coordinator.hpp"

namespace time_utils {
inline bool shutdown_if_deadline_exceeded(
    std::chrono::steady_clock::time_point & last_call_time, std::chrono::milliseconds allowed_threshold, ShutdownCoordinator & shutdown_coordinator)
{
    auto now = std::chrono::steady_clock::now();
    if (last_call_time != std::chrono::steady_clock::time_point{}) {
        auto delta = now - last_call_time;
        if (delta > allowed_threshold) {
            shutdown_coordinator.shutdown(
                std::format("Duration threshold between consecutive callback executions exceeded, allowed threshold={}ms, actual "
                            "elapsed duration={}ms, exiting.",
                    allowed_threshold.count(), std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()));
            return true;
        }
    }
    last_call_time = now;
    return false;
}

// Safely extracts a monotonic publish timestamp from DDS middleware metadata.
// Guards against NTP wall-clock jumps by calculating age and backdating a steady clock.
inline std::chrono::steady_clock::time_point get_safe_monotonic_publish_time(const rclcpp::MessageInfo & message_info, const rclcpp::Logger & logger,
    const std::chrono::steady_clock::time_point & steady_now, const std::chrono::system_clock::time_point & system_now)
{
    int64_t publish_timestamp_ns = message_info.get_rmw_message_info().source_timestamp;
    // Fallback to receive time if source time is missing
    if (publish_timestamp_ns == 0) { publish_timestamp_ns = message_info.get_rmw_message_info().received_timestamp; }
    auto system_publish_time =
        std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(std::chrono::nanoseconds(publish_timestamp_ns));
    auto message_age = system_now - system_publish_time;

    // Prevent NTP backward-jumps from creating negative age
    if (message_age < std::chrono::nanoseconds(0)) {
        RCLCPP_WARN(logger, "Negative computed message age in state callback! Ensure PTP/NTP time is synced.");
        message_age = std::chrono::nanoseconds(0);
    }

    return steady_now - message_age;
}
}  // namespace time_utils