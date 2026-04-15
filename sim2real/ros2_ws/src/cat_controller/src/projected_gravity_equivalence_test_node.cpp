// A minimal ROS 2 node that compares:
//   1) original projected_gravity_body_frame() implementation
//   2) a tf2-based implementation
//
// It runs a deterministic fixed test suite plus randomized normalized quaternions,
// reports exact float equality, tight-tolerance equality, the worst observed error,
// and exits with code 0 on success or 1 on failure.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <vector>

#include "rclcpp/rclcpp.hpp"

class ProjectedGravityEquivalenceTestNode : public rclcpp::Node
{
public:
    ProjectedGravityEquivalenceTestNode() : rclcpp::Node("projected_gravity_equivalence_test_node")
    {
        tolerance_ = this->declare_parameter<double>("tolerance", 1e-6);
        random_test_count_ = this->declare_parameter<int>("random_test_count", 100000);
        log_first_n_mismatches_ = this->declare_parameter<int>("log_first_n_mismatches", 10);
    }

    int run()
    {
        const rclcpp::Logger logger = this->get_logger();

        if (random_test_count_ < 0) {
            RCLCPP_ERROR(logger, "Parameter 'random_test_count' must be >= 0, but got %d", random_test_count_);
            return 1;
        }

        if (tolerance_ < 0.0) {
            RCLCPP_ERROR(logger, "Parameter 'tolerance' must be >= 0, but got %.12g", tolerance_);
            return 1;
        }

        RCLCPP_INFO(logger, "Starting projected gravity equivalence test.");
        RCLCPP_INFO(logger, "Tolerance: %.12g", tolerance_);
        RCLCPP_INFO(logger, "Random test count: %d", random_test_count_);
        RCLCPP_INFO(logger, "Mismatch log limit: %d", log_first_n_mismatches_);

        std::size_t total_test_count = 0;
        std::size_t exact_match_count = 0;
        std::size_t tolerance_match_count = 0;
        std::size_t mismatch_count = 0;
        float worst_absolute_error = 0.0f;
        std::array<float, 4> worst_quaternion_wxyz = {1.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, 3> worst_old_result = {0.0f, 0.0f, -1.0f};
        std::array<float, 3> worst_tf2_result = {0.0f, 0.0f, -1.0f};

        auto process_test_case = [&](const std::array<float, 4> & normalized_quaternion_wxyz, const std::string & label) {
            const std::array<float, 3> old_result = projected_gravity_body_frame_old(normalized_quaternion_wxyz);
            const std::array<float, 3> tf2_result = projected_gravity_body_frame_tf2(normalized_quaternion_wxyz);

            const bool exact_equal = arrays_equal_bitwise(old_result, tf2_result);

            float max_component_abs_error = 0.0f;
            bool within_tolerance = true;
            for (std::size_t index = 0; index < 3; ++index) {
                const float absolute_error = std::fabs(old_result[index] - tf2_result[index]);
                if (absolute_error > static_cast<float>(tolerance_)) { within_tolerance = false; }
                if (absolute_error > max_component_abs_error) { max_component_abs_error = absolute_error; }
            }

            ++total_test_count;
            if (exact_equal) { ++exact_match_count; }
            if (within_tolerance) {
                ++tolerance_match_count;
            } else {
                ++mismatch_count;
                if (mismatch_count <= static_cast<std::size_t>(std::max(0, log_first_n_mismatches_))) {
                    RCLCPP_ERROR(logger,
                        "Mismatch %zu [%s]\n"
                        "  q_wxyz = %s\n"
                        "  old    = %s\n"
                        "  tf2    = %s\n"
                        "  max_abs_error = %.9g",
                        mismatch_count, label.c_str(), format_array(normalized_quaternion_wxyz).c_str(), format_array(old_result).c_str(),
                        format_array(tf2_result).c_str(), static_cast<double>(max_component_abs_error));
                }
            }

            if (max_component_abs_error > worst_absolute_error) {
                worst_absolute_error = max_component_abs_error;
                worst_quaternion_wxyz = normalized_quaternion_wxyz;
                worst_old_result = old_result;
                worst_tf2_result = tf2_result;
            }
        };

        run_fixed_test_cases(process_test_case);
        run_random_test_cases(process_test_case);

        RCLCPP_INFO(logger, "Finished testing.");
        RCLCPP_INFO(logger, "Total tests: %zu", total_test_count);
        RCLCPP_INFO(logger, "Exact float-bit matches: %zu / %zu", exact_match_count, total_test_count);
        RCLCPP_INFO(logger, "Within tolerance: %zu / %zu", tolerance_match_count, total_test_count);
        RCLCPP_INFO(logger, "Tolerance mismatches: %zu", mismatch_count);
        RCLCPP_INFO(logger,
            "Worst max component absolute error: %.9g\n"
            "  q_wxyz = %s\n"
            "  old    = %s\n"
            "  tf2    = %s",
            static_cast<double>(worst_absolute_error), format_array(worst_quaternion_wxyz).c_str(), format_array(worst_old_result).c_str(),
            format_array(worst_tf2_result).c_str());

        if (mismatch_count == 0) {
            RCLCPP_INFO(logger, "PASS: tf2-based implementation matches the original within tolerance.");
            return 0;
        }

        RCLCPP_ERROR(logger, "FAIL: tf2-based implementation does not match the original within tolerance.");
        return 1;
    }

private:
    double tolerance_ = 1e-6;
    int random_test_count_ = 100000;
    int log_first_n_mismatches_ = 10;

    static constexpr float kPi = 3.14159265358979323846f;

    // Original implementation
    static inline std::array<float, 3> projected_gravity_body_frame_old(const std::array<float, 4> & quat_body_to_world_wxyz)
    {
        const float w = quat_body_to_world_wxyz[0];
        const float x = quat_body_to_world_wxyz[1];
        const float y = quat_body_to_world_wxyz[2];
        const float z = quat_body_to_world_wxyz[3];

        const float wi = w;
        const float xi = -x;
        const float yi = -y;
        const float zi = -z;

        // First Hamilton product: q_inv ⊗ g, where g = [0, 0, 0, -1]
        const float a0 = zi;
        const float a1 = -yi;
        const float a2 = xi;
        const float a3 = -wi;

        // Second Hamilton product: (q_inv ⊗ g) ⊗ q
        const float r1 = a0 * x + a1 * w + a2 * z - a3 * y;
        const float r2 = a0 * y - a1 * z + a2 * w + a3 * x;
        const float r3 = a0 * z + a1 * y - a2 * x + a3 * w;

        return {r1, r2, r3};
    }

    // tf2-based implementation
    static inline std::array<float, 3> projected_gravity_body_frame_tf2(const std::array<float, 4> & quat_body_to_world_wxyz)
    {
        // tf2::Quaternion constructor order is (x, y, z, w).
        tf2::Quaternion quaternion_body_to_world_xyzw(
            quat_body_to_world_wxyz[1], quat_body_to_world_wxyz[2], quat_body_to_world_wxyz[3], quat_body_to_world_wxyz[0]);

        quaternion_body_to_world_xyzw.normalize();

        const tf2::Vector3 gravity_world(0.0, 0.0, -1.0);
        const tf2::Vector3 gravity_body = tf2::quatRotate(quaternion_body_to_world_xyzw.inverse(), gravity_world);

        return {static_cast<float>(gravity_body.x()), static_cast<float>(gravity_body.y()), static_cast<float>(gravity_body.z())};
    }

    static std::array<float, 4> normalize_quaternion_wxyz(const std::array<float, 4> & quaternion_wxyz)
    {
        const float w = quaternion_wxyz[0];
        const float x = quaternion_wxyz[1];
        const float y = quaternion_wxyz[2];
        const float z = quaternion_wxyz[3];

        const float squared_norm = w * w + x * x + y * y + z * z;
        if (squared_norm <= std::numeric_limits<float>::min()) { return {1.0f, 0.0f, 0.0f, 0.0f}; }

        const float inverse_norm = 1.0f / std::sqrt(squared_norm);
        return {w * inverse_norm, x * inverse_norm, y * inverse_norm, z * inverse_norm};
    }

    static std::array<float, 4> axis_angle_to_quaternion_wxyz(const std::array<float, 3> & axis_xyz, float angle_rad)
    {
        const float axis_x = axis_xyz[0];
        const float axis_y = axis_xyz[1];
        const float axis_z = axis_xyz[2];

        const float axis_squared_norm = axis_x * axis_x + axis_y * axis_y + axis_z * axis_z;
        if (axis_squared_norm <= std::numeric_limits<float>::min()) { return {1.0f, 0.0f, 0.0f, 0.0f}; }

        const float axis_inverse_norm = 1.0f / std::sqrt(axis_squared_norm);
        const float normalized_axis_x = axis_x * axis_inverse_norm;
        const float normalized_axis_y = axis_y * axis_inverse_norm;
        const float normalized_axis_z = axis_z * axis_inverse_norm;

        const float half_angle = 0.5f * angle_rad;
        const float sin_half_angle = std::sin(half_angle);
        const float cos_half_angle = std::cos(half_angle);

        return {cos_half_angle, normalized_axis_x * sin_half_angle, normalized_axis_y * sin_half_angle, normalized_axis_z * sin_half_angle};
    }

    template <std::size_t element_count>
    static bool arrays_equal_bitwise(const std::array<float, element_count> & left, const std::array<float, element_count> & right)
    { return std::memcmp(left.data(), right.data(), sizeof(float) * element_count) == 0; }

    template <std::size_t element_count>
    static std::string format_array(const std::array<float, element_count> & values)
    {
        std::ostringstream output_stream;
        output_stream << std::fixed << std::setprecision(9) << "[";
        for (std::size_t index = 0; index < element_count; ++index) {
            if (index != 0) { output_stream << ", "; }
            output_stream << values[index];
        }
        output_stream << "]";
        return output_stream.str();
    }

    template <typename ProcessTestCaseFunction>
    void run_fixed_test_cases(ProcessTestCaseFunction process_test_case) const
    {
        const std::vector<std::pair<std::string, std::array<float, 4>>> fixed_cases = {
            {"identity", normalize_quaternion_wxyz({1.0f, 0.0f, 0.0f, 0.0f})},

            {"rot_x_90deg", axis_angle_to_quaternion_wxyz({1.0f, 0.0f, 0.0f}, 0.5f * kPi)},
            {"rot_y_90deg", axis_angle_to_quaternion_wxyz({0.0f, 1.0f, 0.0f}, 0.5f * kPi)},
            {"rot_z_90deg", axis_angle_to_quaternion_wxyz({0.0f, 0.0f, 1.0f}, 0.5f * kPi)},

            {"rot_x_-90deg", axis_angle_to_quaternion_wxyz({1.0f, 0.0f, 0.0f}, -0.5f * kPi)},
            {"rot_y_-90deg", axis_angle_to_quaternion_wxyz({0.0f, 1.0f, 0.0f}, -0.5f * kPi)},
            {"rot_z_-90deg", axis_angle_to_quaternion_wxyz({0.0f, 0.0f, 1.0f}, -0.5f * kPi)},

            {"rot_x_180deg", axis_angle_to_quaternion_wxyz({1.0f, 0.0f, 0.0f}, kPi)},
            {"rot_y_180deg", axis_angle_to_quaternion_wxyz({0.0f, 1.0f, 0.0f}, kPi)},
            {"rot_z_180deg", axis_angle_to_quaternion_wxyz({0.0f, 0.0f, 1.0f}, kPi)},

            {"arbitrary_1", normalize_quaternion_wxyz({0.1825741858f, 0.3651483717f, 0.5477225575f, 0.7302967433f})},
            {"arbitrary_2", normalize_quaternion_wxyz({0.7071067691f, 0.5f, -0.25f, 0.4330126941f})},
            {"arbitrary_3", normalize_quaternion_wxyz({-0.1f, 0.2f, -0.3f, 0.92f})},
            {"arbitrary_4", normalize_quaternion_wxyz({0.0001f, -0.8f, 0.1f, 0.5916f})}};

        for (const auto & test_case : fixed_cases) { process_test_case(test_case.second, test_case.first); }

        // Also explicitly test sign-flipped pairs, since q and -q represent the same rotation.
        for (const auto & test_case : fixed_cases) {
            const std::array<float, 4> q = test_case.second;
            const std::array<float, 4> negative_q = {-q[0], -q[1], -q[2], -q[3]};
            process_test_case(negative_q, test_case.first + "_negated");
        }
    }

    template <typename ProcessTestCaseFunction>
    void run_random_test_cases(ProcessTestCaseFunction process_test_case) const
    {
        std::mt19937 generator(123456789u);
        std::normal_distribution<float> normal_distribution(0.0f, 1.0f);

        for (int test_index = 0; test_index < random_test_count_; ++test_index) {
            std::array<float, 4> quaternion_wxyz = {
                normal_distribution(generator), normal_distribution(generator), normal_distribution(generator), normal_distribution(generator)};
            quaternion_wxyz = normalize_quaternion_wxyz(quaternion_wxyz);

            process_test_case(quaternion_wxyz, "random_" + std::to_string(test_index));

            // Also test the negated quaternion
            const std::array<float, 4> negative_quaternion_wxyz = {
                -quaternion_wxyz[0], -quaternion_wxyz[1], -quaternion_wxyz[2], -quaternion_wxyz[3]};
            process_test_case(negative_quaternion_wxyz, "random_negated_" + std::to_string(test_index));
        }
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    const auto node = std::make_shared<ProjectedGravityEquivalenceTestNode>();
    const int exit_code = node->run();

    rclcpp::shutdown();
    return exit_code;
}