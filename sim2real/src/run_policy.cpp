#include <print>
#include <iostream>
#include <torch/script.h>
#include <memory>

int main(int argc, const char *argv[])
{
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    at::Tensor test = torch::ones({3, 3, 3});

    std::cout << test << std::endl;

    return 0;
}