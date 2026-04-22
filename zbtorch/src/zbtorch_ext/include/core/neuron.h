#pragma once
#include "tensor.h"
#include <memory>
#include <vector>

class Neuron {
public:
    std::vector<std::shared_ptr<Tensor>> w;
    std::shared_ptr<Tensor> b;

    explicit Neuron(int n_inputs);

    [[nodiscard]] std::shared_ptr<Tensor> forward(
        const std::vector<std::shared_ptr<Tensor>>& x) const;

    [[nodiscard]] std::vector<std::shared_ptr<Tensor>> parameters() const;
    void zero_grad() const;
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int n_inputs, int n_outputs);

    [[nodiscard]] std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& x) const;

    [[nodiscard]] std::vector<std::shared_ptr<Tensor>> parameters() const;
    void zero_grad() const;
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(int n_inputs, const std::vector<int>& layer_sizes);

    [[nodiscard]] std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<float>& x) const;

    [[nodiscard]] std::vector<std::shared_ptr<Tensor>> parameters() const;
    void zero_grad() const;
};
