#include <print>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <expected>
#include <fstream>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <tracy/Tracy.hpp>
#define TRACY_NO_CONTEXT_SWITCH
#define TRACY_NO_SYSTEM_TRACING
#define TRACY_NO_VSYNC_CAPTURE
#define TRACY_NO_SAMPLING

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

#include <timed_atomic.hpp>
#include <stamped_robot_state.hpp>

std::shared_ptr<spdlog::logger> logger {nullptr};
const short num_joints = 12;
timed_atomic<stamped_robot_state> global_robot_state {};

std::atomic<bool> exit_flag {false};
static_assert(std::atomic<bool>::is_always_lock_free, "atomic bool is not lock free.");

void exit_handler([[maybe_unused]] int s) {
    exit_flag.store(true);
    logger->error("----------------------------------\nSIGNAL CAUGHT; EXIT FLAG SET!\n------------------------------------");
}

// torch::Tensor construct_observation_tensor() {
//     
//     // TODO: Height map values might need transformation because of how they were defined in isaac lab env
//     // TODO: CONFIRM OBSERVATION SCALE AND ORDER
// }

// Uses timed_atomic to guarantee that each operation only takes a limited amount of time.
// This allows estop / sigint to be noticed below a certain duration.
// Of course this is not proper safety, but this is hard to achieve with the Go2 robot.
void run_control_loop() {
    logger->debug("Starting main control loop.");
    logger->debug("Loading torch model...");
    // TODO: Enable eval mode
    // auto model = load_model();
    auto atomic_op_timeout = std::chrono::microseconds{500};
    auto timeout_threshold = std::chrono::milliseconds{50};

    while(!exit_flag.load()) {
        ZoneScoped;
        FrameMarkNamed("run_control_loop");

        auto robot_state_res = global_robot_state.try_load_for(atomic_op_timeout);
        if(!robot_state_res.has_value()) {
            exit_flag.store(true);
            logger->error("Failed to retrieve robot state within {}us, exiting.", atomic_op_timeout.count());
        }
        auto robot_state = robot_state_res.value();
        // logger->debug("Robot state from control loop: {}", robot_state.timestamp.time_since_epoch().count());
        auto now = std::chrono::steady_clock::now();
        auto delta = now - robot_state.timestamp;
        // logger->debug("robot_state.counter={}\tdelta={}", robot_state.counter, std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
        if(delta > timeout_threshold && robot_state.counter > 0) { // Discard first iteration
            exit_flag.store(true);
            logger->error("State timestamp too old, allowed threshold={}ms, actual state age={}ms. Exiting to prevent outdated states.", 
            timeout_threshold.count(), std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
        }
        // Convert to observation tensor
        std::this_thread::sleep_for(std::chrono::milliseconds{2});
       
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

void append_row_to_csv(const std::string& filename, const std::vector<double>& row) {
    std::ofstream ofs(filename, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Error: could not open " << filename << " for appending\n";
        return;
    }
    for (size_t i = 0; i < row.size(); ++i) {
        ofs << row[i];
        if (i + 1 < row.size())
            ofs << ',';
    }
    ofs << '\n';
}

void robot_state_message_handler(const void *message) {
    ZoneScoped;
    FrameMarkNamed("robot_state_message_handler");
    unitree_go::msg::dds_::LowState_ robot_state = *(unitree_go::msg::dds_::LowState_ *)message;

    static auto last_call_time = std::chrono::steady_clock::time_point{}; // default = epoch
    static constexpr auto timeout_threshold = std::chrono::milliseconds{500};
    static long long iteration_counter = 0;

    auto now = std::chrono::steady_clock::now();
    if (last_call_time != std::chrono::steady_clock::time_point{}) {
        auto delta = now - last_call_time;
        // logger->debug("{}ms", std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
        if (delta > timeout_threshold) {
            exit_flag.store(true);
            logger->error("Duration threshold between consecutive robot state handler callbacks exceeded, allowed threshold={}ms, actual elapsed duration={}ms, exiting.", 
                timeout_threshold.count(), std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
        }
    }
    last_call_time = now;

    auto foot_forces = robot_state.foot_force();
    auto quat_wxyz_body_to_world = robot_state.imu_state().quaternion();
    auto rpy_xyz = robot_state.imu_state().rpy();
    auto projected_gravity = projected_gravity_body_frame(quat_wxyz_body_to_world);
    auto angular_velocity = robot_state.imu_state().gyroscope();

    stamped_robot_state stamped_state;
    stamped_state.foot_forces_raw = foot_forces;
    stamped_state.quat_body_to_world_wxyz = quat_wxyz_body_to_world;
    stamped_state.body_rpy_xyz = rpy_xyz;

    stamped_state.projected_gravity = projected_gravity;
    stamped_state.body_angular_velocity = angular_velocity;
    // Joint order in isaac lab is "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
    // Joint order reported by SDK state array is FR_hip_joint, FR_thigh_joint, FR_calf_joint, FL_hip_joint, FL_thigh_joint, FL_calf_joint, RR_hip_joint, RR_thigh_joint, RR_calf_joint, RL_hip_joint, RL_thigh_joint, RL_calf_joint
    static constexpr int sdk_to_isaac_idx[12] = {
    /*0*/ 1,  // FR_hip → Isaac[1]
    /*1*/ 5,  // FR_thigh → Isaac[5]
    /*2*/ 9,  // FR_calf → Isaac[9]
    /*3*/ 0,  // FL_hip → Isaac[0]
    /*4*/ 4,  // FL_thigh → Isaac[4]
    /*5*/ 8,  // FL_calf → Isaac[8]
    /*6*/ 3,  // RR_hip → Isaac[3]
    /*7*/ 7,  // RR_thigh → Isaac[7]
    /*8*/11,  // RR_calf → Isaac[11]
    /*9*/ 2,  // RL_hip → Isaac[2]
    /*10*/6,  // RL_thigh → Isaac[6]
    /*11*/10  // RL_calf → Isaac[10]
    };

    for (int i = 0; i < num_joints; i++) {
        int j = sdk_to_isaac_idx[i];
        stamped_state.joint_pos[j] = static_cast<float>(robot_state.motor_state()[i].q());
        stamped_state.joint_vel[j] = static_cast<float>(robot_state.motor_state()[i].dq());
    }
    stamped_state.timestamp = now;
    stamped_state.counter = iteration_counter++;
    global_robot_state.try_store_for(stamped_state, std::chrono::microseconds{1000});

    std::vector<double> joint_positions;
    joint_positions.reserve(num_joints);
    for (int i = 0; i < num_joints; i++) {
        joint_positions.push_back(robot_state.motor_state()[i].q());
    }
    // append_row_to_csv("/app/logs/joint_positions.csv", joint_positions);
    logger->debug(
        "Foot forces=[{}]\tIMU RPY=[{:+.4f},{:+.4f},{:+.4f}]\tprojected_gravity=[{:+.4f},{:+.4f},{:.4f}]\tangular_vel=[{:+.4f},{:+.4f},{:+.4f}]\tq=[{}]",
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

    // TODO: Infer input dimension and set global variable

    // run_safety_checklist(robot_state, &robot_state_subscriber);
    run_control_loop();

    // std::this_thread::sleep_for(std::chrono::seconds{10});
    
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