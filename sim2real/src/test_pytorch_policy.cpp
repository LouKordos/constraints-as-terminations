#include <torch/script.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <tracy/Tracy.hpp>

const int observation_dim_no_history = 188;
const int observation_dim_joint_state_history = 236;

int main(int argc, const char* argv[]) {
    std::string checkpoint_path = "/app/traced_checkpoints/2025-06-22-08-06-02_6299_traced_deterministic.pt";
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(checkpoint_path.c_str());
        model.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "Loaded module checkpoint from " << checkpoint_path << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, observation_dim_no_history})); // 1 is for num_envs
    at::Tensor output{};

    for(int i = 0; i < 5; i++) {
        output = model.forward(inputs).toTensor(); // Warmup
    }

    for(int i = 0; i < 10000; i++) {
        ZoneScoped;
        output = model.forward(inputs).toTensor();
    }
    std::cout << output.slice(1,0,5) << std::endl;
    std::cout << "Done.\n";

    return 0;
}