/*
Author: Loukas Kordos
Disclaimer: This code was proudly written without LLMs :)
*/

#include "cat_controller/inference_engine.hpp"

InferenceEngine::InferenceEngine(const std::filesystem::path & checkpoint_path, const int num_joints)
    : num_joints_(num_joints),
      pos_hist_(history_length, num_joints, torch::kCPU),
      vel_hist_(history_length, num_joints, torch::kCPU),
      velocity_command_multiplier_(torch::tensor({2.0f, 2.0f, 0.25f})),
      current_action_(num_joints, 0.0f),
      previous_action_(num_joints, 0.0f)
{
    load_checkpoint(checkpoint_path);

    // This needs to run after loading checkpoint because it depends on the model_observation_dim_
    current_action_.reserve(num_joints);
    previous_action_.reserve(num_joints);
    inference_input_.reserve(1);
    inference_input_.clear();
    inference_input_.push_back(torch::ones({1, model_observation_dim_}));  // To prevent dynamic allocations in loop
}

// Runs inference using the passed observations for a trained policy.
// IMPORTANT: The action joint order is the same as during training and needs conversion to SDK joint order before applying them!
const std::vector<float> & InferenceEngine::generate_action(
    const stamped_robot_state & robot_state, const std::array<float, 3> & vel_command, const std::vector<float> & elevation_map_processed)
{
    torch::InferenceMode guard;  // Almost no cost, thread-local, and disables more than gradguard
    auto observation = construct_observation_tensor(robot_state, vel_command, elevation_map_processed, previous_action_,
        model_observation_dim_ == observation_dim_history, robot_state.counter == 0);
    inference_input_[0] = observation;
    raw_current_action_ = policy_model_.forward(inference_input_).toTensor().contiguous();
    std::memcpy(current_action_.data(), raw_current_action_.data_ptr<float>(), num_joints_ * sizeof(float));
    previous_action_ = current_action_;

    return current_action_;
}

// Can throw exceptions but that's fine if initialized in node constructor.
// This allows RAII where the InferenceEngine (and also the upstream node) should not exist with a failed checkpoint load
void InferenceEngine::load_checkpoint(const std::filesystem::path & checkpoint_path)
{
    try {
        policy_model_ = torch::jit::load(checkpoint_path.string());
        policy_model_.eval();
    } catch (const c10::Error & e) {
        throw std::runtime_error(std::format("Failed to load module, original exception: {}", e.what()));
    }

    int64_t in_features = -1;
    for (const auto & p : policy_model_.named_parameters(/*recurse=*/true)) {
        if (p.name.ends_with(".weight") && p.value.dim() == 2) {
            in_features = p.value.size(1);
            break;
        }
    }

    if (in_features != observation_dim_no_history && in_features != observation_dim_history) {
        throw std::runtime_error(std::format("Observation dimension does not match expected value, exiting. in_features={}", in_features));
    }

    model_observation_dim_ = in_features;
}

torch::Tensor InferenceEngine::construct_observation_tensor(const stamped_robot_state & robot_state, const std::array<float, 3> & vel_command,
    const std::vector<float> & elevation_map_processed, const std::vector<float> & previous_action, bool use_history, bool reset_history)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const int obs_dim = use_history ? observation_dim_history : observation_dim_no_history;
    auto observation = torch::empty({1, obs_dim}, opts);

    // Note that clone() makes the const_cast below safe
    auto base_ang_vel = torch::from_blob(const_cast<float *>(robot_state.body_angular_velocity.data()), {1, 3}, opts).clone();
    base_ang_vel.mul_(0.25f);
    observation.slice(1, 0, 3).copy_(base_ang_vel);

    auto velocity_cmd = torch::from_blob(const_cast<float *>(vel_command.data()), {1, 3}, opts).clone();
    velocity_cmd.mul_(velocity_command_multiplier_);
    observation.slice(1, 3, 6).copy_(velocity_cmd);

    auto projected_gravity = torch::from_blob(const_cast<float *>(robot_state.projected_gravity.data()), {1, 3}, opts).clone();
    projected_gravity.mul_(0.1f);
    observation.slice(1, 6, 9).copy_(projected_gravity);

    if (reset_history) {
        pos_hist_.reset();
        vel_hist_.reset();
    }

    auto jp_cur = torch::from_blob(const_cast<float *>(robot_state.joint_pos.data()), {num_joints_}, opts).clone();
    auto jv_cur = torch::from_blob(const_cast<float *>(robot_state.joint_vel.data()), {num_joints_}, opts).clone();
    jv_cur.mul_(0.05f);

    if (use_history) {
        pos_hist_.update(jp_cur);
        vel_hist_.update(jv_cur);

        const int pos_start = 9;
        const int vel_start = pos_start + history_length * num_joints_;
        observation.slice(1, pos_start, pos_start + history_length * num_joints_).copy_(pos_hist_.flattened().unsqueeze(0));
        observation.slice(1, vel_start, vel_start + history_length * num_joints_).copy_(vel_hist_.flattened().unsqueeze(0));
    } else {
        observation.slice(1, 9, 9 + num_joints_).copy_(jp_cur.unsqueeze(0));
        observation.slice(1, 21, 21 + num_joints_).copy_(jv_cur.unsqueeze(0));
    }

    auto prev_action = torch::from_blob(const_cast<float *>(previous_action.data()), {1, num_joints_}, opts).clone();
    const int prev_action_start_index = use_history ? (9 + 2 * history_length * num_joints_) : (33);
    observation.slice(1, prev_action_start_index, prev_action_start_index + num_joints_).copy_(prev_action);

    const int height_map_start_index = prev_action_start_index + num_joints_;
    const int map_size = static_cast<int>(elevation_map_processed.size());
    // Creates view but lives until end of scope, so copy in next line is sufficient
    auto map_tensor_cpu = torch::from_blob(const_cast<float *>(elevation_map_processed.data()), {1, map_size}, opts);
    observation.slice(1, height_map_start_index, height_map_start_index + map_size).copy_(map_tensor_cpu);

    return observation;
}