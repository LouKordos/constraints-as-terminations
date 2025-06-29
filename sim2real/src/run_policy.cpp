#include <print>
#include <iostream>
#include <memory>
#include <tracy/Tracy.hpp>
#include <thread>
#include <atomic>
#include <mutex>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <torch/script.h>
#include <torch/torch.h>

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/fmt/bundled/ranges.h"

std::shared_ptr<spdlog::logger> logger {nullptr};

std::atomic_flag exit_flag {};
void exit_handler([[maybe_unused]] int s) {
    exit_flag.test_and_set();
    logger->error("----------------------------------\nSIGNAL CAUGHT; EXIT FLAG SET!\n------------------------------------");
}

torch::Tensor get_observation_tensor() {

}

int main([[maybe_unused]] int argc, [[maybe_unused]] const char *argv[])
{
    try {
        ZoneScopedN("Logging init");
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        logger = spdlog::create_async<spdlog::sinks::basic_file_sink_mt>("async_file_logger", "/app/logs/run_policy.log");
        logger->sinks().push_back(console_sink);
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [%l] %v");
        logger->set_level(spdlog::level::debug);
    }
    catch(const spdlog::spdlog_ex &ex) {
        std::cout << "Logging init failed, exiting: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    // SIGINT HANDLER
    struct sigaction sigint_handler;
    sigint_handler.sa_handler = exit_handler;
    sigemptyset(&sigint_handler.sa_mask);
    sigint_handler.sa_flags = 0;
    sigaction(SIGINT, &sigint_handler, NULL);

    // MANY SMALL FUNCTIONS FOR EASY PROFILING AND SEGREGATION!
    // Initial startup in sport mode, calibrate (see safety checklist)
    // Show some states to ensure everything looks good
    // If good, disable sport mode and enter low level control mode
    // Get low level robot state, log / store analytics
    // Convert to observation tensor
    // Run inference to get action
    // Post process action to clip and convert into PD target
    // check target before applying, if it exceeds joint limits, exit
    // Other safety checks from checklist
    // Send low level command with crc32 to robot to execute action
    // Repeat
    
    logger->debug("Reached end of main function, setting exit flag and joining threads...");
    exit_flag.test_and_set();

    // TODO: JOIN ANY THREADS TO ENSURE CLEAN EXIT!
    logger->debug("Flushing and exiting...");
    logger->flush();
    spdlog::shutdown();

    return 0;
}