#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <vector>
#include <tracy/Tracy.hpp>
#include <zmq.hpp>

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
    std::cout << "Starting zeromq server to connect to isaac lab eval script on port 5555" << std::endl;

    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind("tcp://127.0.0.1:5555");

    while(true) {
        zmq::message_t request;
        socket.recv(request, zmq::recv_flags::none); // Blocking receive
        {
            ZoneScopedN("eval_inference");
            const float* data_ptr = static_cast<const float*>(request.data());
            size_t num_floats = request.size() / sizeof(float);

            // Clone to own the data
            torch::Tensor input = torch::from_blob(const_cast<float*>(data_ptr), {1, static_cast<int64_t>(num_floats)}, torch::kFloat32).clone(); // Reshape to (1, N)
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            at::Tensor output = model.forward(inputs).toTensor().contiguous(); // Contiguous for simple memcpy to zmq

            size_t out_bytes = output.numel() * sizeof(float);
            zmq::message_t reply(out_bytes);
            std::memcpy(reply.data(), output.data_ptr(), out_bytes);
            socket.send(reply, zmq::send_flags::none); // Blocking send
        }
    }

    return 0;
}