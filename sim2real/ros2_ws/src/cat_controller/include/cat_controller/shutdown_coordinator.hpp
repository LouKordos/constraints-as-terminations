#pragma once

/*
Author: Loukas Kordos
Disclaimer: This code was proudly written without LLMs :)
*/

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"

/*
This helper coordinates node shutdown:
- Multiple callbacks can run concurrently under a MultiThreadedExecutor and any callback may detect a safety-critical condition and request exit.
- We do not want raw atomics to be manipulated directly throughout the node, because setting an exit flag alone is not enough: timers must be
cancelled, helper processes must be stopped, and the ROS context must be shut down.
- Cleanup must happen exactly once even if multiple callbacks notice the exit condition around the same time.

shutdown(message) is used by anything that wants to request an exit. It sets exit_requested and performs the actual shutdown sequence only once.
handle_exit_if_requested() is used at the top of callbacks so that callbacks which start slightly later can still observe an already-requested
exit and also attempt to shut down the node, without duplicating cleanup.

This keeps the shutdown protocol hard to misuse because callbacks should either call handle_exit_if_requested() at entry, or shutdown(...).
*/
class ShutdownCoordinator
{
public:
    using CleanupCallback = std::function<void()>;
    ShutdownCoordinator(rclcpp::Logger logger, rclcpp::Context::SharedPtr context, CleanupCallback cleanup_callback)
        : logger_(std::move(logger)), context_(std::move(context)), cleanup_callback_(std::move(cleanup_callback))
    { static_assert(std::atomic<bool>::is_always_lock_free, "ShutdownCoordinator requires lock-free atomic<bool>."); }
    ShutdownCoordinator(const ShutdownCoordinator &) = delete;
    ShutdownCoordinator & operator=(const ShutdownCoordinator &) = delete;
    ShutdownCoordinator(ShutdownCoordinator &&) = delete;
    ShutdownCoordinator & operator=(ShutdownCoordinator &&) = delete;

    bool exit_requested() const noexcept { return exit_flag_.load(std::memory_order_acquire); }

    bool handle_exit_if_requested()
    {
        if (!exit_requested()) { return false; }
        shutdown_once(nullptr);
        return true;
    }

    void shutdown(const std::string & message)
    {
        exit_flag_.store(true, std::memory_order_release);
        shutdown_once(&message);
    }

private:
    void shutdown_once(const std::string * message)
    {
        bool expected = false;
        if (!shutdown_started_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) { return; }

        cleanup_callback_();

        if (message != nullptr && !message->empty()) { RCLCPP_ERROR(logger_, "%s", message->c_str()); }
        if (context_) { context_->shutdown("exit flag set"); }
    }

    std::atomic<bool> exit_flag_{false};
    std::atomic<bool> shutdown_started_{false};

    rclcpp::Logger logger_;
    rclcpp::Context::SharedPtr context_;
    CleanupCallback cleanup_callback_;
};