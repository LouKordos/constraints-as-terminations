#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

class Custom
{
public:
    explicit Custom()
    {
    }

    ~Custom()
    {
    }

    void Init();

private:
    void InitLowCmd();
    void LowStateMessageHandler(const void *messages);
    void LowCmdWrite();

private:
    float qInit[3] = {0};
    float qDes[3] = {0};
    float sin_mid_q[3] = {0.0, 1.2, -2.0};
    float Kp[3] = {0};
    float Kd[3] = {0};
    double time_consume = 0;
    int rate_count = 0;
    int sin_count = 0;
    int motiontime = 0;
    float dt = 0.002; // 0.001~0.01

    unitree_go::msg::dds_::LowCmd_ low_cmd{};     // default init
    unitree_go::msg::dds_::LowState_ low_state{}; // default init

    /*publisher*/
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    /*subscriber*/
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;

    /*LowCmd write thread*/
    ThreadPtr lowCmdWriteThreadPtr;
};

uint32_t crc32_core(uint32_t *ptr, uint32_t len)
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

void Custom::Init()
{
    // InitLowCmd();

    /*create publisher*/
    // lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    // lowcmd_publisher->InitChannel();

    /*create subscriber*/
    lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&Custom::LowStateMessageHandler, this, std::placeholders::_1), 1);

    /*loop publishing thread*/
    // lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, 2000, &Custom::LowCmdWrite, this);
}

void Custom::InitLowCmd()
{
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for (int i = 0; i < 20; i++)
    {
        low_cmd.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
        low_cmd.motor_cmd()[i].q() = (PosStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].dq() = (VelStopF);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }
}

std::array<double, 3> quat_to_projected_gravity(const std::array<float, 4> &quat_wxyz_body_to_world)
{
    const double gravity_scale = -0.1; // Form below rotates (0,0,1) but gravity is (0,0,-1)

    // Invert / conjugate because orientation coming from go2 SDK is body => world
    double quat_w_world_to_body = quat_wxyz_body_to_world[0];
    double quat_x_world_to_body = -quat_wxyz_body_to_world[1];
    double quat_y_world_to_body = -quat_wxyz_body_to_world[2];
    double quat_z_world_to_body = -quat_wxyz_body_to_world[3];

    double gravity_x_body = 2.0 * (quat_w_world_to_body * quat_x_world_to_body + quat_y_world_to_body * quat_z_world_to_body) * gravity_scale;
    double gravity_y_body = 2.0 * (quat_w_world_to_body * quat_y_world_to_body - quat_x_world_to_body * quat_z_world_to_body) * gravity_scale;
    double gravity_z_body = (quat_w_world_to_body * quat_w_world_to_body - quat_x_world_to_body * quat_x_world_to_body - quat_y_world_to_body * quat_y_world_to_body + quat_z_world_to_body * quat_z_world_to_body) * gravity_scale;

    return {gravity_x_body, gravity_y_body, gravity_z_body};
}
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

    const float gx = 0.0f, gy = 0.0f, gz = -1.0f;
    // First Hamilton product: q_inv ⊗ g
    const float a0 =  zi;
    const float a1 = -yi;
    const float a2 =  xi;
    const float a3 = -wi;

    // Second Hamilton product: (q_inv ⊗ g) ⊗ q
    const float r0 =  a0*w - a1*x - a2*y - a3*z;  // scalar part (ignored)
    const float r1 =  a0*x + a1*w + a2*z - a3*y;
    const float r2 =  a0*y - a1*z + a2*w + a3*x;
    const float r3 =  a0*z + a1*y - a2*x + a3*w;

    return { r1, r2, r3 };
}

std::array<float, 3> quaternion_to_euler_xyz(const std::array<float, 4> &quat_wxyz)
{
    const float w = quat_wxyz[0];
    const float x = quat_wxyz[1];
    const float y = quat_wxyz[2];
    const float z = quat_wxyz[3];

    // Roll (x‐axis rotation)
    const float sinr_cosp = 2.0f * (w * x + y * z);
    const float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    const float roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (y‐axis rotation)
    const float sinp = 2.0f * (w * y - z * x);
    // clamp to [-1, +1] to avoid nan from asin
    const float pitch = std::asin(std::clamp(sinp, -1.0f, 1.0f));

    // Yaw (z‐axis rotation)
    const float siny_cosp = 2.0f * (w * z + x * y);
    const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    const float yaw = std::atan2(siny_cosp, cosy_cosp);

    return {roll, pitch, yaw};
}

void Custom::LowStateMessageHandler(const void *message)
{
    low_state = *(unitree_go::msg::dds_::LowState_ *)message;
    std::cout << "force=" << low_state.foot_force()[0] << "," << low_state.foot_force()[1] << "," << low_state.foot_force()[2] << "," << low_state.foot_force()[3];
    std::cout << "\t\tIMU: roll,pitch,yaw=" << low_state.imu_state().rpy()[0] << "," << low_state.imu_state().rpy()[1] << "," << low_state.imu_state().rpy()[2];
    auto euler_from_quat = quaternion_to_euler_xyz(low_state.imu_state().quaternion());
    std::cout << "\t\t quat_from_euler roll,pitch,yaw=" << euler_from_quat[0] << "," << euler_from_quat[1] << "," << euler_from_quat[2];
    auto projected_gravity = projected_gravity_body_frame(low_state.imu_state().quaternion());
    std::cout << "\t\tIMU projected gravity x,y,z=" << projected_gravity[0] << "," << projected_gravity[1] << "," << projected_gravity[2];
    std::cout << "\t\tIMU angular vel x,y,z=" << low_state.imu_state().gyroscope()[0] << "," << low_state.imu_state().gyroscope()[1] << "," << low_state.imu_state().gyroscope()[2];
    std::cout << "\t\tjoint angles=";
    for (int i = 0; i < 12; i++)
    {
        std::cout << low_state.motor_state()[i].q();
        if (i != 11)
        {
            std::cout << ",";
        }
    }
    std::cout << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
}

double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos * (1 - rate) + targetPos * rate;
    return p;
}

void Custom::LowCmdWrite()
{
    motiontime++;

    if (motiontime >= 0)
    {
        // first, get record initial position
        if (motiontime >= 0 && motiontime < 20)
        {
            qInit[0] = low_state.motor_state()[0].q();
            qInit[1] = low_state.motor_state()[1].q();
            qInit[2] = low_state.motor_state()[2].q();
        }
        // second, move to the origin point of a sine movement with Kp Kd
        if (motiontime >= 10 && motiontime < 400)
        {
            rate_count++;

            double rate = rate_count / 200.0; // needs count to 200
            Kp[0] = 5.0;
            Kp[1] = 5.0;
            Kp[2] = 5.0;
            Kd[0] = 1.0;
            Kd[1] = 1.0;
            Kd[2] = 1.0;

            qDes[0] = jointLinearInterpolation(qInit[0], sin_mid_q[0], rate);
            qDes[1] = jointLinearInterpolation(qInit[1], sin_mid_q[1], rate);
            qDes[2] = jointLinearInterpolation(qInit[2], sin_mid_q[2], rate);
        }

        double sin_joint1, sin_joint2;
        // last, do sine wave
        float freq_Hz = 1;
        // float freq_Hz = 5;
        float freq_rad = freq_Hz * 2 * M_PI;
        float t = dt * sin_count;

        if (motiontime >= 400)
        {
            sin_count++;
            sin_joint1 = 0.6 * sin(t * freq_rad);
            sin_joint2 = -0.9 * sin(t * freq_rad);
            qDes[0] = sin_mid_q[0];
            qDes[1] = sin_mid_q[1] + sin_joint1;
            qDes[2] = sin_mid_q[2] + sin_joint2;
        }

        low_cmd.motor_cmd()[2].q() = qDes[2];
        low_cmd.motor_cmd()[2].dq() = 0;
        low_cmd.motor_cmd()[2].kp() = Kp[2];
        low_cmd.motor_cmd()[2].kd() = Kd[2];
        low_cmd.motor_cmd()[2].tau() = 0;
    }

    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);

    lowcmd_publisher->Write(low_cmd);
}

int main(int argc, const char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }

    ChannelFactory::Instance()->Init(0, argv[1]);

    Custom custom;
    custom.Init();

    while (1)
    {
        sleep(10);
    }

    return 0;
}
