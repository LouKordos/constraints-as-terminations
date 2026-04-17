#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <string>

#include "cat_controller/stamped_robot_state.hpp"

struct HistoryBuffer
{
    HistoryBuffer(int H, int J, torch::Device dev)
        : buf(torch::zeros({H, J}, torch::TensorOptions().dtype(torch::kFloat32).device(dev))), initialized(false)
    {
    }

    // Insert newest sample (1‑D tensor, shape [J])
    void update(const torch::Tensor & cur)
    {
        buf = torch::roll(buf, {1}, {0});
        buf[0] = cur;
        if (!initialized) {
            for (int i = 1; i < buf.size(0); ++i) { buf[i] = cur; }
            initialized = true;
        }
    }

    // Flatten to [H*J] contiguous, newest first
    torch::Tensor flattened() const { return buf.reshape({-1}).contiguous(); }
    void reset() { initialized = false; }

private:
    torch::Tensor buf;
    bool initialized;
};

class InferenceEngine
{
public:
    InferenceEngine(const std::filesystem::path & checkpoint_path, const int num_joints);
    // Readonly due to const &
    const std::vector<float> & generate_action(
        const stamped_robot_state & robot_state, const std::array<float, 3> & vel_command, const std::vector<float> & elevation_map_processed);
    inline static constexpr int observation_dim_no_history = 188;
    inline static constexpr int observation_dim_history = 236;
    inline static constexpr int history_length = 3;

private:
    torch::Tensor construct_observation_tensor(const stamped_robot_state & robot_state, const std::array<float, 3> & vel_command,
        const std::vector<float> & elevation_map_processed, const std::vector<float> & previous_action, bool use_history, bool reset_history = false);
    void load_checkpoint(const std::filesystem::path & checkpoint_path);

    int num_joints_;
    int model_observation_dim_;
    torch::jit::Module policy_model_;  // Initialized as part of load_checkpoint
    HistoryBuffer pos_hist_;
    HistoryBuffer vel_hist_;
    std::vector<torch::jit::IValue> inference_input_;
    at::Tensor raw_current_action_;
    const torch::Tensor velocity_command_multiplier_;

    // Vector because template would header-only and initializing in constructor means no dynamic heap allocations during inference
    std::vector<float> current_action_;
    std::vector<float> previous_action_;
};