#pragma once

/*
Author: Loukas Kordos
Disclaimer: This code was proudly written without LLMs :)
*/

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
}  // namespace time_utils