#include <print>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <expected>
#include <fstream>
#include <cmath>
#include <ranges>

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
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

// Provide a non-ambiguous overload for streaming std::atomic types.
// Required because otherwise importing torch/script.h produces ambiguous
// overload errors.
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::atomic<T>& v) {
    os << v.load(); 
    return os;
}
#include <torch/script.h>
#include <torch/torch.h>

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/fmt/bundled/ranges.h"

#include <timed_atomic.hpp>
#include <stamped_robot_state.hpp>

std::shared_ptr<spdlog::logger> logger {nullptr};
const short num_joints = 12;
int observation_dim = 188; // TODO: Infer this from loaded model
const float action_scale = 0.8f;
const float actuator_Kp = 25.0f;
const float actuator_Kd = 0.5;
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

// Isaac Lab joint order
const std::array<std::pair<float, float>, 2> base_orientation_limit_rad {std::pair<float, float>{-0.6, 0.6}, {-0.6, 0.6}}; // Only roll and pitch, does not make sense to limit yaw
const std::array<std::pair<float, float>, num_joints> joint_position_limits {std::pair<float, float>{-0.9, 0.9}, {-0.9, 0.9}, {-0.9, 0.9}, {-0.9, 0.9}, {-1.4, 3.4}, {-1.4, 3.4}, {-1.4, 3.4}, {-1.4, 3.4}, {-3, -0.7}, {-3, -0.7}, {-3, -0.7}, {-3, -0.7}}; // rad
std::array<float, num_joints> default_joint_positions {0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5}; // Isaac Lab order
const double joint_vel_abs_limit = 30; // rad/s
const double joint_torque_abs_limit = 40; //Nm
timed_atomic<stamped_robot_state> global_robot_state {};

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

std::atomic<bool> exit_flag {false};
static_assert(std::atomic<bool>::is_always_lock_free, "atomic bool is not lock free.");

void exit_handler([[maybe_unused]] int s) {
    exit_flag.store(true);
    logger->error("----------------------------------\nSIGNAL CAUGHT; EXIT FLAG SET!\n------------------------------------");
}

// Taken from unitree_go2_sdk stand_example
uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}

// Taken from unitree_go2_sdk stand_example
int query_motion_status(unitree::robot::b2::MotionSwitcherClient &msc)
{
    std::string robotForm,motionName;
    int motionStatus;
    int32_t ret = msc.CheckMode(robotForm, motionName);
    if(ret != 0) {
        logger->warn("CheckMode failed. Error code: {}", ret);
    }
    if(motionName.empty())
    {
        motionStatus = 0;
    }
    else
    {
        logger->info("Service {} is still activated...", motionName);
        motionStatus = 1;
    }
    return motionStatus;
}

void enable_low_level_control() {
    logger->debug("Setting up low level control...");
    unitree_go::msg::dds_::LowCmd_ low_cmd{};
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for(int i=0; i<20; i++)
    {
        low_cmd.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
        low_cmd.motor_cmd()[i].q() = (PosStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].dq() = (VelStopF);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }

    std::string robot_command_topic {"rt/lowcmd"};
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(robot_command_topic));
    lowcmd_publisher->InitChannel();

    logger->info("Hold robot using tether or safely put on the floor, low level control mode will be enabled once enter key is registered. This will make robot fall down!");
    std::cin.ignore();

    std::this_thread::sleep_for(std::chrono::seconds{3});

    unitree::robot::b2::MotionSwitcherClient msc;
    msc.SetTimeout(10.0f);
    msc.Init();
    // Shut down motion control-related service
    while(query_motion_status(msc))
    {
        logger->debug("Trying to disable motion control-related service...");
        int32_t ret = msc.ReleaseMode(); 
        if (ret == 0) {
            logger->info("ReleaseMode succeeded.");
        } else {
            logger->error("ReleaseMode failed. Error code: {}", ret);
        }

        logger->debug("Sleeping for 5sec in motion status loop.");
        std::this_thread::sleep_for(std::chrono::seconds{5});
    }

    auto robot_initial_state_res = global_robot_state.try_load_for(std::chrono::microseconds{1000});
    if(!robot_initial_state_res.has_value()) {
        exit_flag.store(true);
        logger->error("Failed to retrieve robot state within {}us, exiting.", std::chrono::microseconds{1000}.count());
    }
    auto initial_robot_state = robot_initial_state_res.value();
    
    // Interpolate to default positions in joint space
    double time = 0.0f;
    double interpolation_duration = 5.0f;
    auto dt = std::chrono::milliseconds{2};
    logger->debug("Starting interpolation to default joint position with initial joint pos (isaac lab order)=[{}]", join_formatted(initial_robot_state.joint_pos));
    for(time = 0.0f; time < interpolation_duration && !exit_flag.load(); time += (dt.count() / 1e+3)) {
        for(int i = 0; i < num_joints; i++) {
            int j = sdk_to_isaac_idx[i];
            low_cmd.motor_cmd()[i].q() = (time/interpolation_duration) * default_joint_positions[j] + (1.0f-time/interpolation_duration) * initial_robot_state.joint_pos[j];
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = actuator_Kp;
            low_cmd.motor_cmd()[i].kd() = actuator_Kd;
            low_cmd.motor_cmd()[i].tau() = 0;
        }
        low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
        lowcmd_publisher->Write(low_cmd);
        logger->debug("t={}\tjoint pos (go2 sdk order)= [{}]", time, fmt::join(low_cmd.motor_cmd() | std::views::take(12) | std::views::transform([](auto &m){ return m.q(); }), ", "));
        std::this_thread::sleep_for(dt); // Run at approximately 500Hz
    }
    logger->info("Finished moving robot to default joint position.");
    while(!exit_flag.load()) {
        lowcmd_publisher->Write(low_cmd);
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

void enable_damping_mode() {
    // Not needed because the robot goes into limp mode after 20ms of no commands anyway
    // Disable low level mode
    // Enable damping mode
}

// Mirrors Isaac Lab ObservationsCfg defined in EnvCfg
torch::Tensor construct_observation_tensor(const stamped_robot_state& robot_state, const std::array<float, 3>& vel_command, const std::array<float, num_joints>& previous_action)
{
    ZoneScoped;
    auto opts = torch::TensorOptions() .dtype(torch::kFloat32).device(torch::kCPU);
    auto observation = at::empty({1, observation_dim}, opts);

    auto base_ang_vel = at::from_blob(const_cast<float*>(robot_state.body_angular_velocity.data()),{1, 3}, opts).clone();
    base_ang_vel.mul_(0.25f);
    observation.slice(1, 0, 3).copy_(base_ang_vel);

    auto velocity_cmd = at::from_blob(const_cast<float*>(vel_command.data()), {1, 3}, opts).clone();
    velocity_cmd.mul_(torch::tensor({2.0f, 2.0f, 0.25f}, opts));
    observation.slice(1, 3, 6).copy_(velocity_cmd);

    auto projected_gravity = at::from_blob(const_cast<float*>(robot_state.projected_gravity.data()), {1, 3}, opts).clone();
    projected_gravity.mul_(0.1f);
    observation.slice(1, 6, 9).copy_(projected_gravity);

    auto joint_positions = at::from_blob(const_cast<float*>(robot_state.joint_pos.data()), {1, 12}, opts).clone();
    observation.slice(1, 9, 9+num_joints).copy_(joint_positions);

    auto joint_velocities = at::from_blob(const_cast<float*>(robot_state.joint_vel.data()), {1, 12}, opts).clone();
    joint_velocities.mul_(0.05f);
    observation.slice(1, 21, 21+num_joints).copy_(joint_velocities);

    auto prev_action = at::from_blob(const_cast<float*>(previous_action.data()), {1, num_joints}, opts).clone();
    observation.slice(1, 33, 33+num_joints).copy_(prev_action);
    // TODO: Height map values might need transformation because of how they were defined in isaac lab env
    // TODO: Replace with real height map data
    observation.slice(1, 33 + num_joints, observation_dim).fill_(-0.33f);

    return observation;
}

// Uses timed_atomic to guarantee that each operation only takes a limited amount of time.
// This allows estop / sigint to be noticed below a certain duration.
// Of course this is not proper safety, but this is hard to achieve with the Go2 robot.
void run_control_loop() {
    logger->debug("Starting main control loop.");
    logger->debug("Loading torch model...");

    // TODO: Make path adjustable/configureable
    std::string checkpoint_path = "/app/traced_checkpoints/2025-06-22-08-06-02_6299_traced_deterministic.pt";
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(checkpoint_path.c_str());
        model.eval();
    }
    catch (const c10::Error& e) {
        logger->error("Failed to load module, exiting.");
        exit_flag.store(true);
    }

    std::cout << "Loaded module checkpoint from " << checkpoint_path << std::endl;
    auto atomic_op_timeout = std::chrono::microseconds{500};
    auto timeout_threshold = std::chrono::milliseconds{50};

    std::array<float, num_joints> current_action {};
    std::array<float, num_joints> previous_action {};
    std::vector<torch::jit::IValue> inference_input;
    inference_input.reserve(1);
    inference_input.clear();
    inference_input.push_back(torch::ones({1, observation_dim})); // To prevent dynamic allocations in loop
    at::Tensor raw_current_action{};
    torch::NoGradGuard no_grad;

    // if(!exit_flag.load()) {
    //     enable_low_level_control();
    //     logger->debug("Enabled low level control mode, entering main control loop");
    //     std::this_thread::sleep_for(std::chrono::seconds{1});
    // }

    // TODO: Make timed loop
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

        // TODO: Get this from teleop
        std::array<float, 3> vel_command {};

        //TODO: Clip vel command and logger->warn that it had to be clipped
        //TODO: Use stamped_vel_command to ensure it's up to date, if it's too old, set to zero and just stand there

        auto observation = construct_observation_tensor(robot_state, vel_command, previous_action);
        inference_input[0] = observation;
        raw_current_action = model.forward(inference_input).toTensor().contiguous();
        std::memcpy(current_action.data(), raw_current_action.data_ptr<float>(), num_joints * sizeof(float));
        previous_action = current_action;
        // logger->debug("raw action={}", current_action);
        // TODO: Store all intermediate values such as current action, pd_targets in rosbag for debugging
        std::array<double, num_joints> pd_target {}; // Go2 native order, NOT Isaac Lab!!!
        for(int i = 0; i < num_joints; i++) {
            int j = sdk_to_isaac_idx[i]; // Remap to go2 order
            pd_target[i] = default_joint_positions[j] + current_action[j] * action_scale; // Scale same as Isaac Lab
        }
        // Do not check if target exceeds joint limits because policy might learn to command out of range values temporarily for more rapid motion.
        if(exit_flag.load()) { // Check before actually applying the action
            logger->error("Exit flag detected in control loop before applying action, exiting.");
            break;
        }
        // Use wrapper with timeout to send low level command with crc32 to robot to execute action
        // std::this_thread::sleep_for(std::chrono::milliseconds{2});
    }
}

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

// Sets exit_flag=true if states are exceeded
void check_state_safety_limits(const stamped_robot_state &robot_state) {
    ZoneScoped;
    if(robot_state.body_rpy_xyz[0] < base_orientation_limit_rad[0].first || robot_state.body_rpy_xyz[0] > base_orientation_limit_rad[0].second) {
        exit_flag.store(true);
        logger->error("Base roll angle out of bounds, roll={}, bounds=[{},{}]", robot_state.body_rpy_xyz[0], base_orientation_limit_rad[0].first, base_orientation_limit_rad[0].second);
    }

    if(robot_state.body_rpy_xyz[1] < base_orientation_limit_rad[1].first || robot_state.body_rpy_xyz[1] > base_orientation_limit_rad[1].second) {
        exit_flag.store(true);
        logger->error("Base pitch angle out of bounds, pitch={}, bounds=[{},{}]", robot_state.body_rpy_xyz[1], base_orientation_limit_rad[1].first, base_orientation_limit_rad[1].second);
    }
    
    for(int i = 0; i < num_joints; i++) {
        if(robot_state.joint_pos[i] < joint_position_limits[i].first || robot_state.joint_pos[i] > joint_position_limits[i].second) {
            exit_flag.store(true);
            logger->error("Joint position for index {} out of bounds, pos={}, bounds=[{},{}]", i, robot_state.joint_pos[i], joint_position_limits[i].first, joint_position_limits[i].second);
        }

        if(std::abs(robot_state.joint_torque[i]) > joint_torque_abs_limit) {
            exit_flag.store(true);
            logger->error("Joint torque for index {} out of bounds, torque={}, limit={}", i, robot_state.joint_torque[i], joint_torque_abs_limit);
        }

        if(std::abs(robot_state.joint_vel[i]) > joint_vel_abs_limit) {
            exit_flag.store(true);
            logger->error("Joint velocity for index {} out of bounds, velocity={}, limit={}", i, robot_state.joint_vel[i], joint_vel_abs_limit);
        }
    }

    // TODO: Put robot into damping mode
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

    for (int i = 0; i < num_joints; i++) {
        int j = sdk_to_isaac_idx[i];
        stamped_state.joint_pos[j] = static_cast<float>(robot_state.motor_state()[i].q());
        stamped_state.joint_vel[j] = static_cast<float>(robot_state.motor_state()[i].dq());
        stamped_state.joint_torque[j] = static_cast<float>(robot_state.motor_state()[i].tau_est());
    }
    stamped_state.timestamp = now;
    stamped_state.counter = iteration_counter++;
    global_robot_state.try_store_for(stamped_state, std::chrono::microseconds{1000});

    check_state_safety_limits(stamped_state);

    if(true) {
        // append_row_to_csv("/app/logs/joint_positions.csv", std::vector<double>(stamped_state.joint_pos.begin(), stamped_state.joint_pos.end()));
        logger->debug(
            "Foot forces=[{}]\tIMU RPY=[{:+.4f},{:+.4f},{:+.4f}]\tprojected_gravity=[{:+.4f},{:+.4f},{:.4f}]\tangular_vel=[{:+.4f},{:+.4f},{:+.4f}]\tq=[{}]",
            fmt::join(foot_forces, ","),
            rpy_xyz[0], rpy_xyz[1], rpy_xyz[2],
            projected_gravity[0], projected_gravity[1], projected_gravity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2],
            join_formatted(stamped_state.joint_pos));   
    }
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

    logger->debug("Finished setting up logging and sigint handler, setting up robot communication.");
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> robot_state_subscriber;
    std::string robot_state_topic {"rt/lowstate"};
    robot_state_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(robot_state_topic));
    robot_state_subscriber->InitChannel(std::bind(&robot_state_message_handler, std::placeholders::_1), 1);

    // TODO: Infer input dimension and set global variable
    run_control_loop();

    // logger->info("Enabling damping mode...");
    // enable_damping_mode();
    
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