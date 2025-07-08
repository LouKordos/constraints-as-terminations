#pragma once
#include <rosbag2_cpp/writer.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <rclcpp/clock.hpp>
#include <atomic>
#include <array>
#include <thread>
#include <chrono>

struct RawSample
{
    std::array<float, 12> joint_pos, joint_vel, joint_tau;
    std::array<float, 3> quat_wxyz;
    std::array<float, 3> body_gyro;
    std::array<float, 3> proj_grav;
    std::array<float, 12> action;
    std::array<float, 12> pd_target;
    std::array<float, 4> foot_force_raw_adc;
    rclcpp::Time stamp;
};

template <std::size_t CAP_P2>
class RingBuffer
{
    static_assert((CAP_P2 & (CAP_P2 - 1u)) == 0, "capacity must be 2^n");

    public:
        bool push(const RawSample &s) noexcept {
            const size_t h = head_.load(std::memory_order_relaxed);
            const size_t nxt = (h + 1) & mask_;
            if (nxt == tail_.load(std::memory_order_acquire))
                return false; // full
            buf_[h] = s;
            head_.store(nxt, std::memory_order_release);
            return true;
        }
        bool pop(RawSample &out) noexcept {
            const size_t t = tail_.load(std::memory_order_relaxed);
            if (t == head_.load(std::memory_order_acquire))
                return false; // empty
            out = buf_[t];
            tail_.store((t + 1) & mask_, std::memory_order_release);
            return true;
        }

    private:
        static constexpr size_t mask_ = CAP_P2 - 1u;
        std::array<RawSample, CAP_P2> buf_;
        std::atomic<size_t> head_{0}, tail_{0};
};

class AsyncRosbagLogger
{
    public:
        static AsyncRosbagLogger &instance()
        {
            static AsyncRosbagLogger i;
            return i;
        }

        void open(const std::string &uri)
        {
            rosbag2_storage::StorageOptions so;
            so.uri = uri;
            so.storage_id = "mcap";
            so.max_cache_size = 8 * 1024 * 1024;
            writer_.open(so);

            js_.name = {"FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"};
            js_.position.resize(12);
            js_.velocity.resize(12);
            js_.effort.resize(12);
            fa_action_.data.resize(12);
            fa_pd_.data.resize(12);

            fa_foot_force_.data.resize(4);
            fa_foot_force_.layout.dim.resize(1);
            fa_foot_force_.layout.dim[0].label  = "foot_order_FL_FR_RL_RR";
            fa_foot_force_.layout.dim[0].size   = 4;
            fa_foot_force_.layout.dim[0].stride = 4;

            run_.store(true, std::memory_order_release);
            worker_ = std::thread(&AsyncRosbagLogger::loop, this);
        }
        void shutdown()
        {
            run_.store(false, std::memory_order_release);
            if (worker_.joinable())
                worker_.join();
        }

        inline void enqueue(const RawSample &s) noexcept { (void)queue_.push(s); }

    private:
        void loop()
        {
            RawSample s;
            while (run_.load(std::memory_order_acquire) || queue_.pop(s))
            {
                while (queue_.pop(s)) {
                    write_sample(s);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
        }
        void write_sample(const RawSample &s)
        {
            // JointState
            js_.header.stamp = s.stamp;
            std::copy(s.joint_pos.begin(), s.joint_pos.end(), js_.position.begin());
            std::copy(s.joint_vel.begin(), s.joint_vel.end(), js_.velocity.begin());
            std::copy(s.joint_tau.begin(), s.joint_tau.end(), js_.effort.begin());
            writer_.write(js_, "/joint_states_isaac_order", js_.header.stamp);

            // IMU
            imu_.header = js_.header;
            imu_.orientation.w = s.quat_wxyz[0];
            imu_.orientation.x = s.quat_wxyz[1];
            imu_.orientation.y = s.quat_wxyz[2];
            imu_.orientation.z = s.quat_wxyz[3];
            imu_.angular_velocity.x = s.body_gyro[0];
            imu_.angular_velocity.y = s.body_gyro[1];
            imu_.angular_velocity.z = s.body_gyro[2];
            imu_.linear_acceleration.x = s.proj_grav[0];
            imu_.linear_acceleration.y = s.proj_grav[1];
            imu_.linear_acceleration.z = s.proj_grav[2];
            writer_.write(imu_, "/imu", imu_.header.stamp);

            std::copy(s.action.begin(), s.action.end(), fa_action_.data.begin());
            std::copy(s.pd_target.begin(), s.pd_target.end(), fa_pd_.data.begin());
            writer_.write(fa_action_, "/policy_action_isaac_order", imu_.header.stamp);
            writer_.write(fa_pd_, "/pd_setpoint_sdk_order", imu_.header.stamp);
            std::copy(s.foot_force_raw_adc.begin(), s.foot_force_raw_adc.end(), fa_foot_force_.data.begin());
            writer_.write(fa_foot_force_, "/foot_force_raw_adc", imu_.header.stamp);
        }

        RingBuffer<1024> queue_; // 2 s buffer at 500 Hz
        rosbag2_cpp::Writer writer_;
        std::thread worker_;
        std::atomic<bool> run_{false};

        // pre‑allocated reusable message instances
        sensor_msgs::msg::JointState js_;
        sensor_msgs::msg::Imu imu_;
        std_msgs::msg::Float32MultiArray fa_action_, fa_pd_, fa_foot_force_;
};