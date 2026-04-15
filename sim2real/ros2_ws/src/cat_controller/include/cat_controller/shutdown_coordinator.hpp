#pragma once

#include <atomic>

class ShutdownCoordinator
{
public:
    ShutdownCoordinator() = default;
    ShutdownCoordinator(const ShutdownCoordinator &) = delete;
    ShutdownCoordinator & operator=(const ShutdownCoordinator &) = delete;
    ShutdownCoordinator(ShutdownCoordinator &&) = delete;
    ShutdownCoordinator & operator=(ShutdownCoordinator &&) = delete;

    void request_exit() noexcept { exit_flag_.store(true, std::memory_order_release); }
    bool exit_requested() const noexcept { return exit_flag_.load(std::memory_order_acquire); }
    bool claim_shutdown_once() noexcept
    {
        bool expected = false;
        return shutdown_started_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
    }

private:
    static_assert(std::atomic<bool>::is_always_lock_free, "ShutdownCoordinator requires lock-free atomic<bool>.");

    std::atomic<bool> exit_flag_{false};
    std::atomic<bool> shutdown_started_{false};
};