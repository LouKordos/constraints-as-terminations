#pragma once

/*
IMPOORTANT NOTE: This is an old class used before switching to ROS and is only kept for the transition to ROS.
Disregard it entirely when looking at the code base, it is deprecated and outdated and should NOT be used!
*/

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <format>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

#include "cat_controller/shutdown_coordinator.hpp"
#include "cat_controller/stamped_robot_state.hpp"
#include "cat_controller/timed_atomic.hpp"

using json = nlohmann::json;

struct LogEntry
{
    bool is_raw;
    std::string json_payload;
};

struct LegacyElevationSample
{
    double ts{0.0};           // Python-side processing time from the legacy JSON packet.
    double map_ts{0.0};       // Source elevation map timestamp from the legacy JSON packet.
    double cpp_recv_ts{0.0};  // Local receive time inside this C++ processor.
    float fill_value{-0.3f};
    std::vector<float> grid;
};

class ElevationMapProcessor
{
public:
    ElevationMapProcessor(std::string log_directory, std::string layer_name, int layer_id, ShutdownCoordinator & shutdown_coordinator,
        rclcpp::Logger logger, const timed_atomic<stamped_robot_state> & robot_state,
        timed_atomic<std::vector<float>> & global_elevation_map_filtered, float hardcoded_elevation = -0.3f, size_t max_recent_samples = 512)
        : zmq_context_(1),
          zmq_socket_(zmq_context_, zmq::socket_type::sub),
          layer_identifier_(layer_id),
          layer_name_(layer_name),
          shutdown_coordinator_(shutdown_coordinator),
          logger_(logger),
          robot_state_(robot_state),
          global_elevation_map_filtered_(global_elevation_map_filtered),
          hardcoded_elevation_(hardcoded_elevation),
          max_recent_samples_(max_recent_samples)
    {
        std::string remote_endpoint = "tcp://192.168.123.224:6973";  // min_filter plugin + relative

        try {
            zmq_socket_.connect(remote_endpoint);
            zmq_socket_.set(zmq::sockopt::subscribe, "");
            RCLCPP_INFO(logger_, "ElevationMapProcessor subscribed to %s for layer '%s'", remote_endpoint.c_str(), layer_name.c_str());
        } catch (const zmq::error_t & e) {
            shutdown_coordinator_.shutdown(std::format("ZMQ Connection failed: {}", e.what()));
            return;
        }

        std::string timestamp_str = std::format("{:%Y-%m-%dT%H-%M-%S}", std::chrono::system_clock::now());

        std::string filename_raw = log_directory + timestamp_str + "_" + layer_name + "_cpp_raw.jsonl";
        file_stream_raw_.open(filename_raw);
        if (!file_stream_raw_.is_open()) {
            shutdown_coordinator_.shutdown(std::format("Critical: Failed to open RAW elevation log file: {}", filename_raw));
            return;
        }

        std::string filename_filtered = log_directory + timestamp_str + "_" + layer_name + "_cpp_filtered.jsonl";
        file_stream_filtered_.open(filename_filtered);
        if (!file_stream_filtered_.is_open()) {
            shutdown_coordinator_.shutdown(std::format("Critical: Failed to open FILTERED elevation log file: {}", filename_filtered));
            return;
        }

        json metadata;
        metadata["type"] = "metadata";
        metadata["version"] = 3;
        metadata["config"] = {{"resolution", elevation_grid_resolution}, {"sensor_offset_x", elevation_sensor_offset_x},
            {"num_x", elevation_grid_width}, {"num_y", elevation_grid_height}, {"fill_value", elevation_fill_value}};
        std::string header = metadata.dump() + "\n";
        file_stream_raw_ << header;
        file_stream_filtered_ << header;
        file_stream_raw_.flush();
        file_stream_filtered_.flush();

        logging_active_.store(true);
        logging_thread_ = std::thread(&ElevationMapProcessor::logging_loop, this);
        processing_thread_ = std::thread(&ElevationMapProcessor::processing_loop, this);
    }

    ~ElevationMapProcessor()
    {
        if (processing_thread_.joinable()) { processing_thread_.join(); }

        logging_active_.store(false);
        logging_cv_.notify_all();

        if (logging_thread_.joinable()) { logging_thread_.join(); }

        if (file_stream_raw_.is_open()) { file_stream_raw_.close(); }
        if (file_stream_filtered_.is_open()) { file_stream_filtered_.close(); }
    }

    std::vector<LegacyElevationSample> get_recent_samples_copy() const
    {
        std::lock_guard<std::mutex> lock(recent_samples_mutex_);
        return std::vector<LegacyElevationSample>(recent_samples_.begin(), recent_samples_.end());
    }

    std::optional<LegacyElevationSample> get_latest_sample_copy() const
    {
        std::lock_guard<std::mutex> lock(recent_samples_mutex_);
        if (recent_samples_.empty()) { return std::nullopt; }
        return recent_samples_.back();
    }

private:
    // Configuration constants
    static constexpr int elevation_grid_width = 13;
    static constexpr int elevation_grid_height = 11;
    static constexpr int elevation_grid_total_size = elevation_grid_width * elevation_grid_height;
    static constexpr float elevation_grid_resolution = 0.08f;
    static constexpr float elevation_sensor_offset_x = 0.2f;
    static constexpr float elevation_fill_value = -0.3f;

    zmq::context_t zmq_context_;
    zmq::socket_t zmq_socket_;
    int layer_identifier_;
    std::string layer_name_;

    ShutdownCoordinator & shutdown_coordinator_;
    rclcpp::Logger logger_;

    // Robot state is const (read-only), map and elevation are mutable (writeable)
    const timed_atomic<stamped_robot_state> & robot_state_;
    timed_atomic<std::vector<float>> & global_elevation_map_filtered_;
    float hardcoded_elevation_;

    std::ofstream file_stream_raw_;
    std::ofstream file_stream_filtered_;

    std::vector<float> raw_data_buffer_ = std::vector<float>(elevation_grid_total_size);

    // Threading primitives declared BEFORE the threads that use them
    std::atomic<bool> logging_active_{false};
    std::queue<LogEntry> log_queue_;
    std::mutex log_mutex_;
    std::condition_variable logging_cv_;

    mutable std::mutex recent_samples_mutex_;
    std::deque<LegacyElevationSample> recent_samples_;
    const size_t max_recent_samples_;

    std::thread processing_thread_;
    std::thread logging_thread_;

    void logging_loop()
    {
        while (logging_active_.load() || !log_queue_.empty()) {
            std::unique_lock<std::mutex> lock(log_mutex_);
            logging_cv_.wait(lock, [this] { return !log_queue_.empty() || !logging_active_.load(); });

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

    void processing_loop()
    {
        auto wait_seconds = std::chrono::seconds(20);
        RCLCPP_INFO(logger_, "ElevationMapProcessor: Waiting %ldsec for external Odom/Mapping pipeline to warm up...", wait_seconds.count());
        auto start_wait = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start_wait < wait_seconds) {
            if (shutdown_coordinator_.exit_requested()) { return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        RCLCPP_INFO(logger_, "ElevationMapProcessor: Warmup complete. Enforcing ZMQ timeouts now.");

        zmq::pollitem_t poll_items[] = {{static_cast<void *>(zmq_socket_), 0, ZMQ_POLLIN, 0}};
        auto safety_timeout = std::chrono::milliseconds{100};

        while (!shutdown_coordinator_.exit_requested()) {
            int rc = zmq::poll(poll_items, 1, safety_timeout);
            if (rc == 0) {
                shutdown_coordinator_.shutdown(
                    std::format("Critical: Elevation map ZMQ timeout (>{}ms). Stopping robot for safety.", safety_timeout.count()));
                continue;
            }

            zmq::message_t zmq_msg;
            if (!zmq_socket_.recv(zmq_msg, zmq::recv_flags::none)) {
                shutdown_coordinator_.shutdown("Critical: ZMQ recv failed.");
                break;
            }

            std::string msg_str(static_cast<char *>(zmq_msg.data()), zmq_msg.size());
            json parsed_json;
            try {
                parsed_json = json::parse(msg_str);
            } catch (const json::parse_error & e) {
                shutdown_coordinator_.shutdown(std::format("Critical: JSON Parse Error: {}", e.what()));
                break;
            }

            float current_fill_val = parsed_json.value("fill_value", elevation_fill_value);

            if (!parsed_json.contains("grid")) {
                shutdown_coordinator_.shutdown("Critical: JSON missing 'grid' field.");
                break;
            }
            raw_data_buffer_ = parsed_json["grid"].get<std::vector<float>>();

            if (raw_data_buffer_.size() != elevation_grid_total_size) {
                shutdown_coordinator_.shutdown(
                    std::format("Critical: Grid size mismatch. Expected {}, got {}", elevation_grid_total_size, raw_data_buffer_.size()));
                break;
            }

            float temporary_elevation_offset = 0.0f;
            for (float & v : raw_data_buffer_) { v += temporary_elevation_offset; }

            auto robot_state_result = robot_state_.try_load_for(std::chrono::microseconds(250));
            if (!robot_state_result.has_value()) {
                shutdown_coordinator_.shutdown("Critical: Timed out trying to load robot state for elevation logging.");
                break;
            }
            stamped_robot_state current_state = robot_state_result.value();

            if (!global_elevation_map_filtered_.try_store_for(raw_data_buffer_, std::chrono::microseconds(250))) {
                shutdown_coordinator_.shutdown("Critical: Timed out trying to update global elevation map atomic.");
                break;
            }

            const double cpp_recv_ts = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

            LegacyElevationSample sample;
            sample.ts = parsed_json.value("ts", 0.0);
            sample.map_ts = parsed_json.value("map_ts", 0.0);
            sample.cpp_recv_ts = cpp_recv_ts;
            sample.fill_value = current_fill_val;
            sample.grid = raw_data_buffer_;
            {
                std::lock_guard<std::mutex> lock(recent_samples_mutex_);
                recent_samples_.push_back(std::move(sample));
                while (recent_samples_.size() > max_recent_samples_) { recent_samples_.pop_front(); }
            }

            json frame_record;
            if (parsed_json.contains("ts")) { frame_record["ts"] = parsed_json["ts"]; }
            if (parsed_json.contains("map_ts")) { frame_record["map_ts"] = parsed_json["map_ts"]; }
            frame_record["cpp_recv_ts"] = cpp_recv_ts;
            frame_record["layer"] = layer_name_;

            int valid_cell_count = 0;
            for (const float & v : raw_data_buffer_) {
                if (std::abs(v - current_fill_val) > 1e-5f) { valid_cell_count++; }
            }
            frame_record["valid"] = static_cast<float>(valid_cell_count) / elevation_grid_total_size;

            float pos_x = 0.0f, pos_y = 0.0f, pos_z = 0.0f;
            if (parsed_json.contains("pose")) {
                pos_x = parsed_json["pose"].value("x", 0.0f);
                pos_y = parsed_json["pose"].value("y", 0.0f);
                pos_z = parsed_json["pose"].value("z", 0.0f);
                RCLCPP_DEBUG(logger_, "Base z: %f", pos_z);
                if (pos_z > 0.8f || pos_z < 0.1f) {
                    shutdown_coordinator_.shutdown("Base z out of safe bounds, likely occluded or odometry wrong, exiting to be safe.");
                }
                hardcoded_elevation_ = -pos_z;
            }

            frame_record["pose"] = {{"x", pos_x}, {"y", pos_y}, {"z", pos_z}, {"qx", current_state.quat_body_to_world_wxyz[1]},
                {"qy", current_state.quat_body_to_world_wxyz[2]}, {"qz", current_state.quat_body_to_world_wxyz[3]},
                {"qw", current_state.quat_body_to_world_wxyz[0]}};

            if (parsed_json.contains("feet")) {
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
                frame_record["grid"] = raw_data_buffer_;

                std::lock_guard<std::mutex> lock(log_mutex_);
                log_queue_.push({false, frame_record.dump()});

                logging_cv_.notify_one();
            }
        }
    }
};