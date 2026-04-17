#pragma once
#include "tensor.h"
#include <memory>
#include <vector>

class Neuron {
public:
    std::vector<std::shared_ptr<Tensor>> w;
    std::shared_ptr<Tensor> b;

    explicit Neuron(int n_inputs);

    std::shared_ptr<Tensor> forward(
        const std::vector<std::shared_ptr<Tensor>>& x) const;

    std::vector<std::shared_ptr<Tensor>> parameters() const;
    void zero_grad();
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int n_inputs, int n_outputs);

    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& x) const;

    std::vector<std::shared_ptr<Tensor>> parameters() const;
    void zero_grad();
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(int n_inputs, const std::vector<int>& layer_sizes);

    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<float>& x) const;

    std::vector<std::shared_ptr<Tensor>> parameters() const;
    void zero_grad();
};
