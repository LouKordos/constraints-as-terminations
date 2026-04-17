/*
This helper is intentionally built as a separate executable and is NOT linked into the ROS node process.

Reason: Unitree's MotionSwitcherClient depends on the SDK2 / CycloneDDS stack that already works in the non-ROS environment, but when the helper
inherits a ROS-sourced environment it can resolve a mixed DDS runtime at startup, e.g. libddsc from /opt/ros/... while libddscxx comes from
/usr/local/lib. That split runtime caused early allocator / ABI crashes such as `free(): invalid pointer` and made the motion switcher unusable from
the ROS process directly.

To avoid that, the main ROS node launches this helper through the dynamic loader with an explicit `--library-path`, so this binary always uses the
known-good DDS libraries from the SDK2 environment. Its only job is to release the high-level motion service via MotionSwitcherClient::ReleaseMode()
and then exit with a clear success / failure status that the ROS node can act on, after which messages published on /lowcmd by the control node are
applied by the robot correctly.
*/

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

int query_motion_status(unitree::robot::b2::MotionSwitcherClient & motion_switcher_client)
{
    std::string robot_form;
    std::string motion_name;

    const int32_t return_code = motion_switcher_client.CheckMode(robot_form, motion_name);
    if (return_code != 0) {
        std::cerr << "CheckMode failed. Error code: " << return_code << std::endl;
        return -1;
    }

    if (motion_name.empty()) { return 0; }

    std::cout << "Service '" << motion_name << "' is still activated..." << std::endl;
    return 1;
}

int main(int argc, const char * argv[])
{
    if (argc < 2) {
        std::cerr << "No network interface specified, usage: " << argv[0] << " [networkInterface]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string network_interface = argv[1];
    constexpr auto overall_timeout = std::chrono::seconds(30);
    const auto deadline = std::chrono::steady_clock::now() + overall_timeout;

    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);
    unitree::robot::b2::MotionSwitcherClient motion_switcher_client;
    motion_switcher_client.SetTimeout(10.0f);
    motion_switcher_client.Init();

    while (std::chrono::steady_clock::now() < deadline) {
        const int motion_status = query_motion_status(motion_switcher_client);

        if (motion_status < 0) { return EXIT_FAILURE; }

        if (motion_status == 0) {
            std::cout << "Motion control service released. Low-level control can now begin." << std::endl;
            return EXIT_SUCCESS;
        }

        std::cout << "Trying to disable sport mode..." << std::endl;
        const int32_t return_code = motion_switcher_client.ReleaseMode();

        if (return_code == 0) {
            std::cout << "ReleaseMode succeeded." << std::endl;
        } else {
            std::cerr << "ReleaseMode failed. Error code: " << return_code << std::endl;
        }

        std::cout << "Sleeping for 3sec in motion status loop." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }

    std::cerr << "Timed out while trying to disable sport mode and trying to enable low level control mode." << std::endl;
    return EXIT_FAILURE;
}