#pragma once
#include <torch/torch.h>

struct HistoryBuffer
{
    HistoryBuffer(int H, int J, torch::Device dev) : buf(torch::zeros({H, J}, torch::TensorOptions().dtype(torch::kFloat32).device(dev))), initialized(false) {}

    // Insert newest sample (1‑D tensor, shape [J])
    void update(const torch::Tensor& cur)
    {
        buf = torch::roll(buf, {1}, {0});
        buf[0] = cur;
        if (!initialized) {
            for (int i = 1; i < buf.size(0); ++i) {
                buf[i] = cur;
            }
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