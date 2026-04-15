#include "cat_controller/low_level_mode_enabler.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstring>
#include <iostream>

LowLevelModeEnabler::LowLevelModeEnabler(std::string helper_binary_path, std::string network_interface, double timeout_seconds)
    : helper_binary_path_(std::move(helper_binary_path)), network_interface_(std::move(network_interface)), timeout_seconds_(timeout_seconds)
{
}

LowLevelModeEnabler::~LowLevelModeEnabler()
{ stop(SIGKILL); }

bool LowLevelModeEnabler::start(std::string & error_message)
{
    if (status_ == Status::Running || status_ == Status::Succeeded) {
        error_message.clear();
        return true;
    }

    const pid_t fork_result = fork();
    if (fork_result < 0) {
        failure_message_ = std::string("fork() failed while starting motion switcher helper: ") + std::strerror(errno);
        status_ = Status::Failed;
        error_message = failure_message_;
        return false;
    }

    if (fork_result == 0) {
        const char * loader_path = "/lib64/ld-linux-x86-64.so.2";
        const char * library_path = "/usr/local/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu";

        execl(loader_path, loader_path, "--library-path", library_path, helper_binary_path_.c_str(), network_interface_.c_str(),
            static_cast<char *>(nullptr));
        std::cerr << "execl() failed for release_motion_mode via ld-linux: " << std::strerror(errno) << std::endl;
        _exit(127);
    }

    child_pid_ = fork_result;
    start_time_ = std::chrono::steady_clock::now();
    status_ = Status::Running;
    failure_message_.clear();
    error_message.clear();
    return true;
}

LowLevelModeEnabler::Status LowLevelModeEnabler::poll(std::string & error_message)
{
    if (status_ == Status::Failed) {
        error_message = failure_message_;
        return status_;
    }

    if (status_ == Status::Succeeded || status_ == Status::NotStarted) {
        error_message.clear();
        return status_;
    }

    if (child_pid_ <= 0) {
        failure_message_ = "Motion switcher helper process was never started correctly.";
        status_ = Status::Failed;
        error_message = failure_message_;
        return status_;
    }

    int wait_status = 0;
    const pid_t wait_result = waitpid(child_pid_, &wait_status, WNOHANG);

    if (wait_result == 0) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_seconds = std::chrono::duration<double>(now - start_time_).count();

        if (elapsed_seconds > timeout_seconds_) {
            stop(SIGKILL);
            failure_message_ = "Timed out waiting for motion switcher helper to enable low-level control mode.";
            status_ = Status::Failed;
            error_message = failure_message_;
            return status_;
        }

        error_message.clear();
        return status_;
    }

    if (wait_result < 0) {
        failure_message_ = std::string("waitpid() failed while monitoring motion switcher helper: ") + std::strerror(errno);
        status_ = Status::Failed;
        error_message = failure_message_;
        return status_;
    }

    child_pid_ = -1;

    if (WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0) {
        status_ = Status::Succeeded;
        error_message.clear();
        return status_;
    }

    if (WIFEXITED(wait_status)) {
        failure_message_ = "Motion switcher helper exited with failure code " + std::to_string(WEXITSTATUS(wait_status)) +
                           ". Low-level control mode was not enabled.";
        status_ = Status::Failed;
        error_message = failure_message_;
        return status_;
    }

    if (WIFSIGNALED(wait_status)) {
        failure_message_ =
            "Motion switcher helper was terminated by signal " + std::to_string(WTERMSIG(wait_status)) + ". Low-level control mode was not enabled.";
        status_ = Status::Failed;
        error_message = failure_message_;
        return status_;
    }

    failure_message_ = "Motion switcher helper ended in an unknown state.";
    status_ = Status::Failed;
    error_message = failure_message_;
    return status_;
}

void LowLevelModeEnabler::stop(int signal_number)
{
    if (child_pid_ <= 0) { return; }

    kill(child_pid_, signal_number);
    int wait_status = 0;
    waitpid(child_pid_, &wait_status, 0);
    child_pid_ = -1;

    if (status_ == Status::Running) { status_ = Status::Failed; }
}