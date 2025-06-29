#include <print>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <expected>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <tracy/Tracy.hpp>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>

// #include <torch/script.h>
// #include <torch/torch.h>

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/fmt/bundled/ranges.h"

std::shared_ptr<spdlog::logger> logger {nullptr};
const short num_joints = 12;

std::atomic<bool> exit_flag {false};
static_assert(std::atomic<bool>::is_always_lock_free(), "atomic bool is not lock free.");

void exit_handler([[maybe_unused]] int s) {
    exit_flag.store(true);
    logger->error("----------------------------------\nSIGNAL CAUGHT; EXIT FLAG SET!\n------------------------------------");
}

// torch::Tensor construct_observation_tensor() {
//     // TODO: Infer input dimension to return joint state history or not
//     // TODO: Height map values might need transformation because of how they were defined in isaac lab env
//     // TODO: CONFIRM OBSERVATION SCALE AND ORDER
// }

// Uses timed_atomic to guarantee that each operation only takes a limited amount of time.
// This allows estop / sigint to be noticed below a certain duration.
// Of course this is not proper safety, but this is hard to achieve with the Go2 robot.
void run_control_loop() {
    logger->debug("Starting main control loop.");

    // auto model = load_model();

    while(!exit_flag.load()) {
    // MANY SMALL FUNCTIONS FOR EASY PROFILING AND SEGREGATION!
    // Get low level robot state
    // Convert to observation tensor
    // TODO: Enable eval mode
    // Run inference to get action
    // Post process action to clip and convert into PD target
    // check target before applying, if it exceeds joint limits, exit
    // Other safety checks from checklist
    // Check exit flag
    // Use wrapper with timeout to send low level command with crc32 to robot to execute action
    // Repeat
    }
}

// void run_safety_checklist(const unitree_go::msg::dds_::LowState_ &robot_state, const unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> &robot_state_subscriber) {
//     logger->debug("Starting initial safety checklist...");
//     // Initial startup in sport mode, calibrate (see safety checklist)
//     // Show some states to ensure everything looks good
//     // If good, disable sport mode and enter low level control mode => return
//     // If bad, set exit flag and return
//     // TODO: Detailed logging
// }

// TODO: Move into helper class or replace with library
// Compute body-frame gravity vector given a body→world quaternion.
// quat_body_to_world_wxyz is a unit quaternion [w, x, y, z] rotating body to world.
// Returns [g_x, g_y, g_z] in body frame.
static inline std::array<float, 3> projected_gravity_body_frame(const std::array<float, 4> &quat_body_to_world_wxyz)
{
    // Extract components
    const float w = quat_body_to_world_wxyz[0];
    const float x = quat_body_to_world_wxyz[1];
    const float y = quat_body_to_world_wxyz[2];
    const float z = quat_body_to_world_wxyz[3];

    const float wi =  w;
    const float xi = -x;
    const float yi = -y;
    const float zi = -z;

    // First Hamilton product: q_inv ⊗ g
    const float a0 =  zi;
    const float a1 = -yi;
    const float a2 =  xi;
    const float a3 = -wi;

    // Second Hamilton product: (q_inv ⊗ g) ⊗ q
    const float r1 =  a0*x + a1*w + a2*z - a3*y;
    const float r2 =  a0*y - a1*z + a2*w + a3*x;
    const float r3 =  a0*z + a1*y - a2*x + a3*w;

    return { r1, r2, r3 };
}

// Helper: format each element with `fmt_spec` and join with `sep`
template<typename Range>
std::string join_formatted(const Range& values, std::string_view fmt_spec = "{:.4f}", std::string_view sep = ",")
{
    std::vector<std::string> formatted;
    formatted.reserve(std::size(values));
    for (auto&& v : values) {
        formatted.push_back(fmt::format(fmt::runtime(fmt_spec), v));
    }

    return fmt::format("{}", fmt::join(formatted, sep));
}

void robot_state_message_handler(const void *message) {
    ZoneScoped;
    FrameMark;
    unitree_go::msg::dds_::LowState_ robot_state = *(unitree_go::msg::dds_::LowState_ *)message;

    auto foot_forces = robot_state.foot_force();
    auto rpy_xyz = robot_state.imu_state().rpy();
    auto projected_gravity = projected_gravity_body_frame(robot_state.imu_state().quaternion());
    auto angular_velocity = robot_state.imu_state().gyroscope();
    std::vector<double> joint_positions;
    joint_positions.reserve(num_joints);
    for (int i = 0; i < num_joints; i++) {
        joint_positions.push_back(robot_state.motor_state()[i].q());
    }
    logger->debug(
        "Foot forces=[{}]\tIMU RPY=[{:.4f},{:.4f},{:.4f}]\tprojected_gravity=[{:.4f},{:.4f},{:.4f}]\tangular_vel=[{:.4f},{:.4f},{:.4f}]\tq=[{}]",
        fmt::join(foot_forces, ","),
        rpy_xyz[0], rpy_xyz[1], rpy_xyz[2],
        projected_gravity[0], projected_gravity[1], projected_gravity[2],
        angular_velocity[0], angular_velocity[1], angular_velocity[2],
        join_formatted(joint_positions));
    // std::this_thread::sleep_for(std::chrono::milliseconds{200});
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

    if (argc < 2)
    {
        logger->error("No network interface specified, usage: {} [networkInterface]", argv[0]);
        return EXIT_FAILURE;
    }

    // SIGINT HANDLER
    struct sigaction sigint_handler;
    sigint_handler.sa_handler = exit_handler;
    sigemptyset(&sigint_handler.sa_mask);
    sigint_handler.sa_flags = 0;
    sigaction(SIGINT, &sigint_handler, NULL);

    logger->debug("Finished setting up loggin and sigint handler, setting up robot communication.");
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> robot_state_subscriber;
    std::string robot_state_topic {"rt/lowstate"};
    robot_state_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(robot_state_topic));
    robot_state_subscriber->InitChannel(std::bind(&robot_state_message_handler, std::placeholders::_1), 1);

    // run_safety_checklist(robot_state, &robot_state_subscriber);
    // run_control_loop();

    std::this_thread::sleep_for(std::chrono::seconds{10});
    
    logger->debug("Reached end of main function, setting exit flag and joining threads...");
    exit_flag.store(true);

    logger->debug("Closing robot state subscriber channel...");
    robot_state_subscriber->CloseChannel();
    logger->debug("Closed robot state subscriber channel.");

    // TODO: JOIN ANY THREADS TO ENSURE CLEAN EXIT!
    logger->debug("Flushing and exiting...");
    logger->flush();
    spdlog::shutdown();

    return 0;
}