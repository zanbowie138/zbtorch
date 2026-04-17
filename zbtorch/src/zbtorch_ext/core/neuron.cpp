#include <zbtorch_ext/neuron.h>
#include <random>
#include <stdexcept>

static float random_uniform_11() {
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(rng);
}

// ---------------------------------------------------------------------------
// Neuron
// ---------------------------------------------------------------------------
Neuron::Neuron(int n_inputs) {
    w.reserve(n_inputs);
    for (int i = 0; i < n_inputs; ++i)
        w.push_back(std::make_shared<Tensor>(random_uniform_11()));
    b = std::make_shared<Tensor>(0.0f);
}

std::shared_ptr<Tensor> Neuron::forward(
    const std::vector<std::shared_ptr<Tensor>>& x) const
{
    if (x.size() != w.size())
        throw std::runtime_error("Neuron::forward: input size mismatch");

    if (w.empty())
        return std::make_shared<Tensor>(b->tanh());

    // Add first product to b so b is a real node in the graph (not a copy).
    // Every intermediate must be heap-allocated before the next operator call
    // because Tensor operators internally call shared_from_this().
    auto acc = std::make_shared<Tensor>(*w[0] * *x[0]);
    acc = std::make_shared<Tensor>(*acc + *b);
    for (size_t i = 1; i < w.size(); ++i) {
        auto product = std::make_shared<Tensor>(*w[i] * *x[i]);
        acc = std::make_shared<Tensor>(*acc + *product);
    }
    return std::make_shared<Tensor>(acc->tanh());
}

std::vector<std::shared_ptr<Tensor>> Neuron::parameters() const {
    std::vector<std::shared_ptr<Tensor>> params(w.begin(), w.end());
    params.push_back(b);
    return params;
}

void Neuron::zero_grad() {
    for (auto& p : w)
        p->grad.assign(p->data.size(), 0.0f);
    b->grad.assign(b->data.size(), 0.0f);
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------
Layer::Layer(int n_inputs, int n_outputs) {
    neurons.reserve(n_outputs);
    for (int i = 0; i < n_outputs; ++i)
        neurons.emplace_back(n_inputs);
}

std::vector<std::shared_ptr<Tensor>> Layer::forward(
    const std::vector<std::shared_ptr<Tensor>>& x) const
{
    std::vector<std::shared_ptr<Tensor>> out;
    out.reserve(neurons.size());
    for (const auto& n : neurons)
        out.push_back(n.forward(x));
    return out;
}

std::vector<std::shared_ptr<Tensor>> Layer::parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (const auto& n : neurons) {
        auto np = n.parameters();
        params.insert(params.end(), np.begin(), np.end());
    }
    return params;
}

void Layer::zero_grad() {
    for (auto& n : neurons)
        n.zero_grad();
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------
MLP::MLP(int n_inputs, const std::vector<int>& layer_sizes) {
    if (layer_sizes.empty())
        throw std::runtime_error("MLP: layer_sizes must not be empty");
    layers.reserve(layer_sizes.size());
    int in = n_inputs;
    for (int out : layer_sizes) {
        layers.emplace_back(in, out);
        in = out;
    }
}

std::vector<std::shared_ptr<Tensor>> MLP::forward(
    const std::vector<float>& x) const
{
    std::vector<std::shared_ptr<Tensor>> current;
    current.reserve(x.size());
    for (float xi : x)
        current.push_back(std::make_shared<Tensor>(xi));
    for (const auto& layer : layers)
        current = layer.forward(current);
    return current;
}

std::vector<std::shared_ptr<Tensor>> MLP::parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (const auto& layer : layers) {
        auto lp = layer.parameters();
        params.insert(params.end(), lp.begin(), lp.end());
    }
    return params;
}

void MLP::zero_grad() {
    for (auto& layer : layers)
        layer.zero_grad();
}
