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
#include <format>
#include <filesystem>
#include <sstream>
#include <cstdint>
#include <cstring>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <tracy/Tracy.hpp>
// #define TRACY_NO_CONTEXT_SWITCH
#define TRACY_NO_SYSTEM_TRACING
#define TRACY_NO_VSYNC_CAPTURE
// #define TRACY_NO_SAMPLING

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/HeightMap_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

// #include <sensor_msgs/msg/joint_state.hpp>
// #include <sensor_msgs/msg/imu.hpp>
// #include <std_msgs/msg/float32_multi_array.hpp>
// #include <rclcpp/clock.hpp>

#include <zmq.hpp>

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
// #include <async_rosbag_logger.hpp>
#include <history_buffer.hpp>

std::shared_ptr<spdlog::logger> logger {nullptr};
const short num_joints = 12;
int observation_dim_no_history = 188;
int observation_dim_history = 236;
int history_length = 3;
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

constexpr auto vel_cmd_stale_threshold = std::chrono::milliseconds{200};
constexpr auto vel_cmd_zmq_poll_timeout = std::chrono::milliseconds{300};
timed_atomic<std::array<float, 3>> global_vel_command { {0.0f, 0.0f, 0.0f} };

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

std::atomic<bool> exit_flag {false};
static_assert(std::atomic<bool>::is_always_lock_free, "atomic bool is not lock free.");

void exit_handler([[maybe_unused]] int s) {
    exit_flag.store(true);
    logger->error("----------------------------------\nSIGNAL CAUGHT; EXIT FLAG SET!\n------------------------------------");
}

// --- Elevation Map Configuration ---
const int elevation_grid_width = 13;
const int elevation_grid_height = 11;
const int elevation_grid_total_size = elevation_grid_width * elevation_grid_height;
const float elevation_grid_resolution = 0.08f;
const float elevation_sensor_offset_x = 0.2f;
const float elevation_fill_value = -0.27f; 
const float hardcoded_elevation = -0.30f;

timed_atomic<std::vector<float>> global_elevation_map_filtered{
    std::vector<float>(elevation_grid_total_size, hardcoded_elevation)
};

struct LogEntry {
    bool is_raw;
    std::string json_payload;
};

class ElevationMapProcessor {
public:
    ElevationMapProcessor(std::string log_directory, std::string layer_name, int layer_id) 
        : zmq_context_(1), zmq_socket_(zmq_context_, zmq::socket_type::sub), layer_identifier_(layer_id), layer_name_(layer_name)
    {
        std::string remote_endpoint = "tcp://192.168.123.224:6975"; 
        
        try {
            zmq_socket_.connect(remote_endpoint);
            zmq_socket_.set(zmq::sockopt::subscribe, ""); 
            logger->info("ElevationMapProcessor subscribed to {} for layer '{}'", remote_endpoint, layer_name);
        } catch (const zmq::error_t& e) {
            logger->error("ZMQ Connection failed: {}", e.what());
            exit_flag.store(true);
            return;
        }

        std::string timestamp_str = std::format("{:%Y-%m-%dT%H-%M-%S}", std::chrono::system_clock::now());
        
        std::string filename_raw = log_directory + timestamp_str + "_" + layer_name + "_cpp_raw.jsonl";
        file_stream_raw_.open(filename_raw);
        if (!file_stream_raw_.is_open()) {
            logger->error("Critical: Failed to open RAW elevation log file: {}", filename_raw);
            exit_flag.store(true); return;
        }

        std::string filename_filtered = log_directory + timestamp_str + "_" + layer_name + "_cpp_filtered.jsonl";
        file_stream_filtered_.open(filename_filtered);
        if (!file_stream_filtered_.is_open()) {
            logger->error("Critical: Failed to open FILTERED elevation log file: {}", filename_filtered);
            exit_flag.store(true); return;
        }

        json metadata;
        metadata["type"] = "metadata";
        metadata["version"] = 3;
        metadata["config"] = {
            {"resolution", elevation_grid_resolution},
            {"sensor_offset_x", elevation_sensor_offset_x},
            {"num_x", elevation_grid_width},
            {"num_y", elevation_grid_height},
            {"fill_value", elevation_fill_value}
        };
        std::string header = metadata.dump() + "\n";
        file_stream_raw_ << header;
        file_stream_filtered_ << header;
        file_stream_raw_.flush();
        file_stream_filtered_.flush();

        logging_active_ = true;
        logging_thread_ = std::thread(&ElevationMapProcessor::logging_loop, this);
        processing_thread_ = std::thread(&ElevationMapProcessor::processing_loop, this);
    }

    ~ElevationMapProcessor() {
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }

        logging_active_ = false;
        logging_cv_.notify_all();
        if (logging_thread_.joinable()) {
            logging_thread_.join();
        }

        if (file_stream_raw_.is_open()) file_stream_raw_.close();
        if (file_stream_filtered_.is_open()) file_stream_filtered_.close();
    }

private:
    zmq::context_t zmq_context_;
    zmq::socket_t zmq_socket_;
    int layer_identifier_;
    std::string layer_name_;
    
    std::ofstream file_stream_raw_;
    std::ofstream file_stream_filtered_;
    
    std::thread processing_thread_;
    std::thread logging_thread_;
    
    std::vector<float> raw_data_buffer_ = std::vector<float>(elevation_grid_total_size);
    std::vector<float> filtered_data_buffer_ = std::vector<float>(elevation_grid_total_size);

    std::atomic<bool> logging_active_{false};
    std::queue<LogEntry> log_queue_;
    std::mutex log_mutex_;
    std::condition_variable logging_cv_;

    void apply_masked_spatial_filter(const std::vector<float>& input, std::vector<float>& output, float current_fill_val) {
        for (int y = 0; y < elevation_grid_height; ++y) {
            for (int x = 0; x < elevation_grid_width; ++x) {
                int current_idx = y * elevation_grid_width + x;
                double val = static_cast<double>(input[current_idx]);

                if (std::abs(val - current_fill_val) < 1e-5) {
                    output[current_idx] = current_fill_val;
                    continue;
                }

                double accumulator = 0.0;
                int valid_neighbor_count = 0;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ny = y + dy;
                        int nx = x + dx;

                        if (nx >= 0 && nx < elevation_grid_width && ny >= 0 && ny < elevation_grid_height) {
                            double neighbor_val = static_cast<double>(input[ny * elevation_grid_width + nx]);
                            if (std::abs(neighbor_val - current_fill_val) > 1e-5) {
                                accumulator += neighbor_val;
                                valid_neighbor_count++;
                            }
                        }
                    }
                }

                output[current_idx] = (valid_neighbor_count > 0) ? static_cast<float>(accumulator / valid_neighbor_count) : static_cast<float>(val);
            }
        }
    }

    void logging_loop() {
        while (logging_active_ || !log_queue_.empty()) {
            std::unique_lock<std::mutex> lock(log_mutex_);
            logging_cv_.wait(lock, [this] { return !log_queue_.empty() || !logging_active_; });

            while (!log_queue_.empty()) {
                LogEntry entry = std::move(log_queue_.front());
                log_queue_.pop();
                
                lock.unlock(); 
                
                if (entry.is_raw && file_stream_raw_.is_open()) {
                    file_stream_raw_ << entry.json_payload << "\n";
                    file_stream_raw_.flush(); 
                } else if (!entry.is_raw && file_stream_filtered_.is_open()) {
                    file_stream_filtered_ << entry.json_payload << "\n";
                    file_stream_filtered_.flush();
                }

                lock.lock();
            }
        }
    }

    void processing_loop() {
        auto wait_seconds = std::chrono::seconds(50);
        logger->info("ElevationMapProcessor: Waiting {}sec for external Odom/Mapping pipeline to warm up...", wait_seconds.count());
        auto start_wait = std::chrono::steady_clock::now();
        while(std::chrono::steady_clock::now() - start_wait < wait_seconds) { if(exit_flag.load()) {return;} std::this_thread::sleep_for(std::chrono::milliseconds(100)); } 
        logger->info("ElevationMapProcessor: Warmup complete. Enforcing ZMQ timeouts now."); 
        
        zmq::pollitem_t poll_items[] = { { static_cast<void*>(zmq_socket_), 0, ZMQ_POLLIN, 0 } };
        auto safety_timeout = std::chrono::milliseconds{500};

        while (!exit_flag.load()) {
            ZoneScopedN("ElevationProcessingLoop");
            
            int rc = zmq::poll(poll_items, 1, safety_timeout);
            if (rc == 0) {
                exit_flag.store(true);
                logger->error("Critical: Elevation map ZMQ timeout (>500ms). Stopping robot for safety.");
                continue;
            }

            zmq::message_t zmq_msg;
            if (!zmq_socket_.recv(zmq_msg, zmq::recv_flags::none)) {
                exit_flag.store(true);
                logger->error("Critical: ZMQ recv failed.");
                break;
            }

            std::string msg_str(static_cast<char*>(zmq_msg.data()), zmq_msg.size());
            json parsed_json;
            try {
                parsed_json = json::parse(msg_str);
            } catch (const json::parse_error& e) {
                exit_flag.store(true);
                logger->error("Critical: JSON Parse Error: {}", e.what());
                break;
            }

            float current_fill_val = parsed_json.value("fill_value", elevation_fill_value);

            if(!parsed_json.contains("grid")) {
                exit_flag.store(true);
                logger->error("Critical: JSON missing 'grid' field.");
                break;
            }
            raw_data_buffer_ = parsed_json["grid"].get<std::vector<float>>();
            
            if (raw_data_buffer_.size() != elevation_grid_total_size) {
                 exit_flag.store(true);
                 logger->error("Critical: Grid size mismatch. Expected {}, got {}", elevation_grid_total_size, raw_data_buffer_.size());
                 break;
            }

            float temporary_elevation_offset = 0.0; 
            for(float &v : raw_data_buffer_) {
                v += temporary_elevation_offset;
            }

            auto robot_state_result = global_robot_state.try_load_for(std::chrono::microseconds(100));
            if (!robot_state_result.has_value()) {
                 exit_flag.store(true);
                 logger->error("Critical: Failed to load robot state for elevation logging.");
                 break;
            }
            stamped_robot_state current_state = robot_state_result.value();
            
            const bool apply_filter = false;
            if(apply_filter) {
                apply_masked_spatial_filter(raw_data_buffer_, filtered_data_buffer_, current_fill_val);
            }
            else {
                filtered_data_buffer_ = raw_data_buffer_;
            }

            if(!global_elevation_map_filtered.try_store_for(filtered_data_buffer_, std::chrono::microseconds(100))) {
                logger->error("Critical: Failed to update global elevation map atomic.");
                exit_flag.store(true);
                break;
            }

            json frame_record;
            if(parsed_json.contains("ts")) frame_record["ts"] = parsed_json["ts"];
            frame_record["cpp_recv_ts"] = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            frame_record["layer"] = layer_name_;

            int valid_cell_count = 0;
            for (const float &v : filtered_data_buffer_) {
                if (std::abs(v - current_fill_val) > 1e-5) valid_cell_count++;
            }
            frame_record["valid"] = static_cast<float>(valid_cell_count) / elevation_grid_total_size;

            float pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
            if(parsed_json.contains("pose")) {
                pos_x = parsed_json["pose"].value("x", 0.0f);
                pos_y = parsed_json["pose"].value("y", 0.0f);
                pos_z = parsed_json["pose"].value("z", 0.0f);
            }
            
            frame_record["pose"] = {
                {"x", pos_x}, {"y", pos_y}, {"z", pos_z},
                {"qx", current_state.quat_body_to_world_wxyz[1]},
                {"qy", current_state.quat_body_to_world_wxyz[2]},
                {"qz", current_state.quat_body_to_world_wxyz[3]},
                {"qw", current_state.quat_body_to_world_wxyz[0]}
            };

            if(parsed_json.contains("feet")) {
                frame_record["feet"] = parsed_json["feet"]; 
            } else {
                frame_record["feet"] = nullptr;
            }

            {
                json raw_frame = frame_record;
                raw_frame["grid"] = raw_data_buffer_;
                
                std::lock_guard<std::mutex> lock(log_mutex_);
                log_queue_.push({true, raw_frame.dump()});
            }

            {
                frame_record["grid"] = filtered_data_buffer_;
                
                std::lock_guard<std::mutex> lock(log_mutex_);
                log_queue_.push({false, frame_record.dump()});
                
                logging_cv_.notify_one();
            }
        }
    }
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
        logger->warn("CheckMode failed, exiting. Error code: {}", ret);
        exit_flag.store(true);
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

// TODO: Move to class to prevent global access and only allow sending commands via timed_atomic!
unitree_go::msg::dds_::LowCmd_ low_cmd{};
unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
unitree::common::ThreadPtr lowCmdWriteThreadPtr;
timed_atomic<std::array<float, num_joints>> pd_setpoint_sdk_order {};
timed_atomic<std::array<float, num_joints>> global_current_action_isaac_order {};
auto atomic_op_timeout = std::chrono::microseconds{500};

void send_pd_commands() {
    ZoneScoped;
    FrameMark;
    if(exit_flag.load()) {
        return; // Will cause robot to disable because no commands have been received
    }

    auto setpoint_res = pd_setpoint_sdk_order.try_load_for(atomic_op_timeout);
    if(!setpoint_res.has_value()) {
        exit_flag.store(true);
        logger->error("Failed to fetch desired action within {}us in send_pd_commands(), exiting.", atomic_op_timeout.count());
        return;
    }
    auto setpoint_sdk_order = setpoint_res.value();

    for(int i = 0; i < num_joints; i++) {
        low_cmd.motor_cmd()[i].q() = setpoint_sdk_order[i];
        low_cmd.motor_cmd()[i].dq() = 0;
        low_cmd.motor_cmd()[i].kp() = actuator_Kp;
        low_cmd.motor_cmd()[i].kd() = actuator_Kd;
        low_cmd.motor_cmd()[i].tau() = 0;
    }

    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
    lowcmd_publisher->Write(low_cmd);
}

void enable_low_level_control() {
    logger->debug("Setting up low level control...");

    // Init commands
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
    lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(robot_command_topic));
    lowcmd_publisher->InitChannel();

    logger->info("Hold robot using tether or safely put on the floor, low level control mode will be enabled once enter key is registered. This will make robot fall down!");
    std::cin.ignore();
    logger->info("Sleeping for 10sec to let user reevaluate his choices :)");
    std::this_thread::sleep_for(std::chrono::seconds{10});

    unitree::robot::b2::MotionSwitcherClient msc;
    msc.SetTimeout(10.0f);
    msc.Init();
    // Shut down motion control-related service
    while(query_motion_status(msc) && !exit_flag.load())
    {
        logger->debug("Trying to disable motion control-related service...");
        int32_t ret = msc.ReleaseMode(); 
        if (ret == 0) {
            logger->info("ReleaseMode succeeded.");
        } else {
            logger->error("ReleaseMode failed. Error code: {}", ret);
        }

        logger->debug("Sleeping for 5sec in motion status loop.");
        std::this_thread::sleep_for(std::chrono::seconds{3});
    }

    // Start sending commands to robot at 500Hz
    lowCmdWriteThreadPtr = unitree::common::CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, 2000, &send_pd_commands);

    auto robot_initial_state_res = global_robot_state.try_load_for(std::chrono::microseconds{1000});
    if(!robot_initial_state_res.has_value()) {
        exit_flag.store(true);
        logger->error("Failed to retrieve robot state within {}us, exiting.", std::chrono::microseconds{1000}.count());
        return;
    }
    stamped_robot_state initial_robot_state = robot_initial_state_res.value();

    // Linear interpolation to default joint position, policy actions are offsets from these default positions
    float interpolation_duration = 5.0f;
    auto dt = std::chrono::milliseconds{2};
    std::array<float, num_joints> temp_setpoint_sdk_order {};
    for(float interpolation_time = 0.0f; interpolation_time < interpolation_duration && !exit_flag.load(); interpolation_time += (dt.count() / 1e+3)) {
        for(int i = 0; i < num_joints; i++) {
            int j = sdk_to_isaac_idx[i];
            temp_setpoint_sdk_order[i] = std::min((interpolation_time/interpolation_duration), 1.0f) * default_joint_positions[j] + std::max((1.0f-interpolation_time/interpolation_duration), 0.0f) * initial_robot_state.joint_pos[j];
        }
        logger->debug("t={:.3f}\tjoint pos (go2 sdk order)= [{}]", interpolation_time, join_formatted(temp_setpoint_sdk_order));
        if(!pd_setpoint_sdk_order.try_store_for(temp_setpoint_sdk_order, atomic_op_timeout)) {
            exit_flag.store(true);
            logger->error("Failed to update global PD setpoint within {}us during linear interpolation to default joint positions, exiting.", atomic_op_timeout.count());
        }
        std::this_thread::sleep_for(dt);
    }

    logger->info("Finished linear interpolation to default joint positions, low level command mode activation succeeded. Sleeping for 3sec to stabilize...");
    std::this_thread::sleep_for(std::chrono::seconds{3});
}

// TODO for when deadline is not haunting you: Move this into class
torch::Tensor construct_observation_tensor(const stamped_robot_state& robot_state, const std::array<float, 3>& vel_command, const std::array<float, num_joints>& previous_action, bool use_history, bool reset_history = false)
{
    ZoneScoped;
    auto opts  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const int obs_dim = use_history ? observation_dim_history : observation_dim_no_history;
    auto observation = torch::empty({1, obs_dim}, opts);

    auto base_ang_vel = torch::from_blob(const_cast<float*>(robot_state.body_angular_velocity.data()), {1, 3}, opts).clone();
    base_ang_vel.mul_(0.25f);
    observation.slice(1, 0, 3).copy_(base_ang_vel);

    auto velocity_cmd = torch::from_blob(const_cast<float*>(vel_command.data()), {1, 3}, opts).clone();
    velocity_cmd.mul_(torch::tensor({2.0f, 2.0f, 0.25f}, opts));
    observation.slice(1, 3, 6).copy_(velocity_cmd);

    auto projected_gravity = torch::from_blob(const_cast<float*>(robot_state.projected_gravity.data()), {1, 3}, opts).clone();
    projected_gravity.mul_(0.1f);
    observation.slice(1, 6, 9).copy_(projected_gravity);

    static HistoryBuffer pos_hist(history_length, num_joints, torch::kCPU);
    static HistoryBuffer vel_hist(history_length, num_joints, torch::kCPU);
    if (reset_history) { pos_hist.reset(); vel_hist.reset(); }

    auto jp_cur = torch::from_blob(const_cast<float*>(robot_state.joint_pos.data()), {num_joints}, opts).clone();
    auto jv_cur = torch::from_blob(const_cast<float*>(robot_state.joint_vel.data()), {num_joints}, opts).clone();
    jv_cur.mul_(0.05f);

    if (use_history) {
        pos_hist.update(jp_cur);
        vel_hist.update(jv_cur);

        const int pos_start = 9;
        const int vel_start = pos_start + history_length * num_joints;
        observation.slice(1, pos_start, pos_start + history_length * num_joints).copy_(pos_hist.flattened().unsqueeze(0));
        observation.slice(1, vel_start, vel_start + history_length * num_joints).copy_(vel_hist.flattened().unsqueeze(0));
    }
    else {
        observation.slice(1, 9, 9 + num_joints).copy_(jp_cur.unsqueeze(0));
        observation.slice(1, 21, 21 + num_joints).copy_(jv_cur.unsqueeze(0));
    }

    auto prev_action = torch::from_blob(const_cast<float*>(previous_action.data()), {1, num_joints}, opts).clone();
    const int prev_action_start_index = use_history ? (9 + 2 * history_length * num_joints) : (33);
    observation.slice(1, prev_action_start_index, prev_action_start_index + num_joints).copy_(prev_action);

    const int height_map_start_index = prev_action_start_index + num_joints;
    const bool use_hardcoded_heights = true;
    if(use_hardcoded_heights) {
        observation.slice(1, height_map_start_index, obs_dim).fill_(hardcoded_elevation);
    }
    else {
        auto elevation_data_result = global_elevation_map_filtered.try_load_for(std::chrono::microseconds(50));
        if (elevation_data_result.has_value()) {
            std::vector<float> map_vector = elevation_data_result.value();
            auto map_tensor_cpu = torch::from_blob(map_vector.data(), {1, elevation_grid_total_size}, opts); // Creates view but lives until end of scope, so copy in next line is sufficient
            observation.slice(1, height_map_start_index, height_map_start_index + elevation_grid_total_size).copy_(map_tensor_cpu);
        } else {
            logger->error("Critical: Failed to load global elevation map in observation construction.");
            exit_flag.store(true);
            observation.slice(1, height_map_start_index, height_map_start_index + elevation_grid_total_size).fill_(elevation_fill_value);
        }
    }

    return observation;
}

// Uses timed_atomic to guarantee that each operation only takes a limited amount of time.
// This allows estop / sigint to be noticed below a certain duration.
// Of course this is not proper safety, but this is hard to achieve with the Go2 robot.
void run_control_loop(std::filesystem::path checkpoint_path, std::filesystem::path logdir_path) {
    logger->debug("Starting main control loop.");
    logger->debug("Loading torch model...");

    torch::jit::Module model;
    try {
        model = torch::jit::load(checkpoint_path.string());
        model.eval();
        logger->debug("Successfully loaded traced model.");
    }
    catch (const c10::Error& e) {
        logger->error("Failed to load module, exiting.");
        exit_flag.store(true);
    }

    int64_t in_features = -1;
    for (const auto& p : model.named_parameters(/*recurse=*/true)) {
        if (p.name.ends_with(".weight") && p.value.dim() == 2) {
            in_features = p.value.size(1);
            break;
        }
    }
    if(in_features != observation_dim_no_history && in_features != observation_dim_history) {
        logger->error("Observation dimension does not match expected value, exiting. in_features={}", in_features);
        exit_flag.store(true);
    }

    int model_observation_dim = in_features;
    logger->info("Loaded module checkpoint from {} with observation dimension={}", checkpoint_path.string(), model_observation_dim);
    auto state_timeout_threshold = std::chrono::milliseconds{50};

    std::array<float, num_joints> current_action {};
    std::array<float, num_joints> previous_action {};
    std::vector<torch::jit::IValue> inference_input;
    inference_input.reserve(1);
    inference_input.clear();
    inference_input.push_back(torch::ones({1, model_observation_dim})); // To prevent dynamic allocations in loop
    at::Tensor raw_current_action{};
    torch::NoGradGuard no_grad;

    if(!exit_flag.load()) {
        enable_low_level_control();
        logger->debug("Enabled low level control mode, entering main control loop");
        std::this_thread::sleep_for(std::chrono::seconds{1});
    }

	std::array<float, 3> vel_command_mag_limit = {2.0, 2.0, 1.0}; //vel_x, vel_y, omega_z    

    // We start the thread here because the 25sec inside processing_loop should only start after the startup procedure is done
    std::unique_ptr<ElevationMapProcessor> elevation_processor = std::make_unique<ElevationMapProcessor>(
        logdir_path.string(),
        "smooth",
        2 // Min Filter Layer (ID 1)
    );

    auto dt = std::chrono::milliseconds{20};
    auto frequency_overrun_threshold = std::chrono::milliseconds{10};
    auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto next_iteration_time = std::chrono::steady_clock::now();
    static long long iteration_counter = 0;

    while(!exit_flag.load()) {
        ZoneScoped;
        FrameMarkNamed("run_control_loop");
        next_iteration_time += dt;

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
        if(delta > state_timeout_threshold && robot_state.counter > 0) { // Discard first iteration
            exit_flag.store(true);
            logger->error("State timestamp too old, allowed threshold={}ms, actual state age={}ms. Exiting to prevent outdated states.", 
            state_timeout_threshold.count(), std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
        }

        std::array<float, 3> vel_command {0.0, 0.0, 0.0};
        if(auto vcmd = global_vel_command.try_load_for(atomic_op_timeout); vcmd.has_value()) {
            vel_command = vcmd.value();
        }
        else {
            logger->error("Failed to fetch vel_command within {}us, exiting.", atomic_op_timeout.count());
            exit_flag.store(true);
        }
        for(int i = 0; i < 3; i++) {
            if(std::abs(vel_command[i]) > vel_command_mag_limit[i]) {
                logger->warn("Had to clip vel_command[{}]={}, vel_command_mag_limit[i]={}", i, vel_command[i], vel_command_mag_limit[i]);
                vel_command[i] = std::max(-vel_command_mag_limit[i], std::min(vel_command_mag_limit[i], vel_command[i]));
            }	
        }

        const bool walk_a_bit = false;
        if(walk_a_bit) {logger->warn("ROBOT WILL WALK SOON!");}
        auto time_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if(walk_a_bit && time_now_ms - start_ms > 55000 && time_now_ms - start_ms < 60000) {
            vel_command[0] = 1.0f;
        }

        auto observation = construct_observation_tensor(robot_state, vel_command, previous_action, model_observation_dim == observation_dim_history, robot_state.counter == 0);
        inference_input[0] = observation;
        {ZoneScopedN("Inference (model.forward)");
            raw_current_action = model.forward(inference_input).toTensor().contiguous();
        }
        std::memcpy(current_action.data(), raw_current_action.data_ptr<float>(), num_joints * sizeof(float));
        if(!global_current_action_isaac_order.try_store_for(current_action, atomic_op_timeout)) {
            logger->error("Failed to set global current_action within {}us, exiting.", atomic_op_timeout.count());
            exit_flag.store(true);
        }
        // logger->debug("raw action={}", current_action);
        auto delta_action = [](auto const& curr, auto const& prev, double dt){ 
            std::array<double, num_joints> r; 
            std::ranges::transform(curr, prev, r.begin(), 
                [dt](auto a, auto b){return (a-b)/dt;}); return r; }(current_action, previous_action, dt.count() / 1e+3);
        // logger->debug("delta_action/dt= [{}]", join_formatted(delta_action));
        // TODO: Store all intermediate values such as current action, pd_targets in rosbag for debugging
        std::array<float, num_joints> pd_target_sdk_order {}; // Go2 native order, NOT Isaac Lab!!!
        for(int i = 0; i < num_joints; i++) {
            int j = sdk_to_isaac_idx[i]; // Remap to go2 order
            pd_target_sdk_order[i] = default_joint_positions[j] + current_action[j] * action_scale; // Scale same as Isaac Lab
        }
        // Do not check if target exceeds joint limits because policy might learn to command out of range values temporarily for more rapid motion.
        if(exit_flag.load()) { // Check before actually applying the action
            logger->error("Exit flag detected in control loop before applying action, exiting.");
            break;
        }

        if(!pd_setpoint_sdk_order.try_store_for(pd_target_sdk_order, atomic_op_timeout)) {
            exit_flag.store(true);
            logger->error("Failed to update global PD target within {}us, exiting.", atomic_op_timeout.count());
        }
        previous_action = current_action;
        iteration_counter++;
        // Scoped to exclude in tracy profiling
        {ZoneScopedN("SleepUntilNextCycle");
            auto loop_end_time = std::chrono::steady_clock::now();
            if(loop_end_time < next_iteration_time) {
                std::this_thread::sleep_until(next_iteration_time);
            }
            else {
                auto overrun = std::chrono::duration_cast<std::chrono::microseconds>(loop_end_time - next_iteration_time);
                logger->error("Control deadline exceeded by {}us!", overrun.count());
                if(overrun > frequency_overrun_threshold && iteration_counter > 5) { // Warm up first 5 iterations
                    exit_flag.store(true);
                    logger->error("Control loop frequency overrun exceeds threshold of {}ms, exiting for safety...", frequency_overrun_threshold.count());
                }
            }
        }
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

// DEPRECATED
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
    global_robot_state.try_store_for(stamped_state, atomic_op_timeout);

    check_state_safety_limits(stamped_state);

    if(false) {
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

void height_map_handler(const void *message) {
    unitree_go::msg::dds_::HeightMap_ height_map = *(unitree_go::msg::dds_::HeightMap_*)message;

    // logger->debug("Heightmap res={}\twidth={}\theight={}\torigin_x={}\torigin_y={}", height_map.resolution(), height_map.width(), height_map.height(), height_map.origin()[0], height_map.origin()[1]);
}

void vel_command_listener(std::string endpoint)
{
    zmq::context_t ctx{1};
    zmq::socket_t sock{ctx, zmq::socket_type::pull};

    try {
        sock.bind(endpoint);
        logger->info("Velocity command listener connected to {}", endpoint);
    }
    catch(const zmq::error_t& e) {
        logger->error("Failed to connect PULL socket ({}), zeroing commands and exiting.", e.what());
        exit_flag.store(true);
        global_vel_command.try_store_for({0.0f,0.0f,0.0f}, atomic_op_timeout);
        return;
    }

    zmq::pollitem_t poll_items[] = { { static_cast<void*>(sock), 0, ZMQ_POLLIN, 0 } };
    std::array<float, 3> zero {0.0f,0.0f,0.0f};

    while(!exit_flag.load()) {
        ZoneScopedN("vel_command_listener_loop");
        int rc = zmq::poll(poll_items, 1, vel_cmd_zmq_poll_timeout);
        if(rc == 0) {
            logger->warn("No vel command received for {}ms, setting global command to zero", vel_cmd_zmq_poll_timeout.count());
            global_vel_command.try_store_for(zero, atomic_op_timeout);
            continue;
        }

        zmq::message_t msg;
        if(!sock.recv(msg, zmq::recv_flags::none)) {
            logger->warn("Failed to recv() on velocity command socket, keeping previous value.");
            continue;
        }

        std::string_view payload{static_cast<char*>(msg.data()), msg.size()};
        std::stringstream ss{std::string(payload)};
        long long ts_ms;
        float vx, vy, omega;
        char comma;

        if(!(ss >> ts_ms >> comma >> vx >> comma >> vy >> comma >> omega)) {
            logger->warn("Malformed velocity command: '{}' Storing zero.", payload);
            global_vel_command.try_store_for(zero, atomic_op_timeout);
            continue;
        }

        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto age = now_ms - ts_ms;

        if(age > vel_cmd_stale_threshold.count()) {
            logger->warn("Stale velocity command (age {} ms > {} ms), zeroing.", age, vel_cmd_stale_threshold.count());
            global_vel_command.try_store_for(zero, atomic_op_timeout);
            continue;
        }

        global_vel_command.try_store_for({vx, vy, omega}, atomic_op_timeout);
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] const char *argv[])
{
    // Forces all standard libs to use UTC
    setenv("TZ", "", 1);
    tzset();

    auto run_timestamp_utc = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::filesystem::path logdir_path {"/app/logs/utc_" + std::format("{:%Y-%m-%d-%H-%M-%S}", run_timestamp_utc) + "/"};
    std::error_code ec;

    if(std::filesystem::create_directories(logdir_path, ec)) {
        std::cout << "Successfully created logdir at " << logdir_path << std::endl;
    }
    else {
        if(ec) {
            std::cerr << "Error creating logdir: " << ec.message() << " (code " << ec.value() << "), exiting.\n";
            exit_flag.store(true);
            return EXIT_FAILURE;
        } // else path already existed
    }

    try {
        ZoneScopedN("Logging init");
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        logger = spdlog::create_async<spdlog::sinks::basic_file_sink_mt>("async_file_logger", logdir_path.string() + "run_policy.log");
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

    // std::filesystem::path checkpoint_path {"/app/traced_checkpoints/2025-06-22-08-06-02_6299_traced_deterministic.pt"};
    std::filesystem::path checkpoint_path {"/app/traced_checkpoints/2025-06-28-17-13-04_21349_traced_deterministic.pt"};
    logger->info("Using checkpoint at {}", checkpoint_path.string());

    logger->debug("Setting up robot communication.");
    // AsyncRosbagLogger::instance().open((logdir_path / "rosbag").string());
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> robot_state_subscriber;
    std::string robot_state_topic {"rt/lowstate"};
    robot_state_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(robot_state_topic));
    robot_state_subscriber->InitChannel(std::bind(&robot_state_message_handler, std::placeholders::_1), 1);

    // unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::HeightMap_> height_map_subscriber;
    // std::string height_map_topic {"rt/utlidar/height_map_array"};
    // height_map_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::HeightMap_>(height_map_topic));
    // height_map_subscriber->InitChannel(std::bind(&height_map_handler, std::placeholders::_1), 1);

    std::string zmq_endpoint = "tcp://*:6969";
	std::thread vel_cmd_listener_thread(vel_command_listener, zmq_endpoint);

    std::this_thread::sleep_for(std::chrono::milliseconds{500});

    auto state_res = global_robot_state.try_load_for(atomic_op_timeout);
    if(!state_res.has_value()) {
        exit_flag.store(true);
        logger->error("Failed to fetch state within {}us in main(), exiting.", atomic_op_timeout.count());
    }

    // Set to current position to prevent sudden jumps when low level controller is enabled
    std::array<float, num_joints> temp_setpoint {};
    for(int i = 0; i < num_joints; i++) {
        temp_setpoint[i] = state_res.value().joint_pos[sdk_to_isaac_idx[i]];
    }
    if(!pd_setpoint_sdk_order.try_store_for(temp_setpoint, atomic_op_timeout)) {
        exit_flag.store(true);
        logger->error("Failed to set PD setpoint within {}us in main(), exiting.", atomic_op_timeout.count());
    }

    // TODO: Infer input dimension and set global variable
    run_control_loop(checkpoint_path, logdir_path);

    // logger->info("Enabling damping mode...");
    // enable_damping_mode();
    
    logger->debug("Reached end of main function, setting exit flag and joining threads...");
    exit_flag.store(true);

    logger->debug("Closing robot channels...");
    robot_state_subscriber->CloseChannel();
    // height_map_subscriber->CloseChannel();
    lowcmd_publisher->CloseChannel();
    logger->debug("Closed robot channels.");

    if(vel_cmd_listener_thread.joinable()) vel_cmd_listener_thread.join();
    // AsyncRosbagLogger::instance().shutdown();
    logger->debug("Flushing logs...");
    logger->flush();
    spdlog::shutdown();
    std::cout << "Spdlog shut down, exiting. Terminal may freeze for a bit due to tracy-capture processing events in the background" << std::endl;

    return 0;
}

