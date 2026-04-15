#pragma once

#include <sys/types.h>

#include <chrono>
#include <string>

class LowLevelModeEnabler
{
public:
    enum class Status
    {
        NotStarted,
        Running,
        Succeeded,
        Failed
    };

    LowLevelModeEnabler(std::string helper_binary_path, std::string network_interface, double timeout_seconds);

    ~LowLevelModeEnabler();

    bool start(std::string & error_message);
    Status poll_thread_unsafe(std::string & error_message);
    void stop(int signal_number = 9);

private:
    std::string helper_binary_path_;
    std::string network_interface_;
    double timeout_seconds_;

    pid_t child_pid_ = -1;
    Status status_ = Status::NotStarted;
    std::string failure_message_;
    std::chrono::steady_clock::time_point start_time_;
};