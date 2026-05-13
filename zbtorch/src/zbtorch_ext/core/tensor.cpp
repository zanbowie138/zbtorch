#include <core/tensor.h>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape)
        throw std::runtime_error("Tensor shape mismatch");
}

static void check_same_device(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device())
        throw std::runtime_error(
            "Tensors must be on the same device; use .to() to move");
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

Tensor::Tensor()
    : _backward([](const float*, int) {}) {}

Tensor::Tensor(float scalar, Device device, std::string label)
    : shape({1}), _label(std::move(label)),
      _backward([](const float*, int) {}) {
    data_ = make_storage(1, device);
    grad_ = make_storage(1, device);
    get_ops(device).fill(data_.data(), 1, scalar);
    get_ops(device).zero(grad_.data(), 1);
}

Tensor::Tensor(const std::vector<float>& d, const std::vector<size_t>& s,
               Device device, std::string label)
    : shape(s), _label(std::move(label)),
      _backward([](const float*, int) {}) {
    int n = static_cast<int>(d.size());
    // Build on CPU first, then transfer if needed.
    Storage cpu_buf = make_cpu_storage(n);
    std::copy(d.begin(), d.end(), cpu_buf.data());
    data_ = copy_storage(cpu_buf, device);
    grad_ = make_storage(n, device);
    get_ops(device).zero(grad_.data(), n);
}

Tensor::Tensor(const Tensor& other)
    : shape(other.shape), _op(other._op), _label(other._label),
      _children(other._children), _backward(other._backward) {
    if (other.data_.valid())
        data_ = copy_storage(other.data_, other.data_.device);
    if (other.grad_.valid())
        grad_ = copy_storage(other.grad_, other.grad_.device);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    shape     = other.shape;
    _op       = other._op;
    _label    = other._label;
    _children = other._children;
    _backward = other._backward;
    if (other.data_.valid())
        data_ = copy_storage(other.data_, other.data_.device);
    if (other.grad_.valid())
        grad_ = copy_storage(other.grad_, other.grad_.device);
    return *this;
}

// ---------------------------------------------------------------------------
// Device transfer
// ---------------------------------------------------------------------------

Tensor Tensor::to(Device d) const {
    Tensor out;
    out.shape  = shape;
    out._label = _label;
    if (data_.valid()) out.data_ = copy_storage(data_, d);
    if (grad_.valid()) out.grad_ = copy_storage(grad_, d);
    return out;
}

// ---------------------------------------------------------------------------
// Host-side accessors
// ---------------------------------------------------------------------------

std::vector<float> Tensor::cpu_data() const {
    if (!data_.valid()) return {};
    if (data_.device != CPU)
        throw std::runtime_error("cpu_data: tensor is on a non-CPU device");
    return std::vector<float>(data_.data(), data_.data() + data_.size);
}

std::vector<float> Tensor::cpu_grad() const {
    if (!grad_.valid()) return {};
    if (grad_.device != CPU)
        throw std::runtime_error("cpu_grad: tensor is on a non-CPU device");
    return std::vector<float>(grad_.data(), grad_.data() + grad_.size);
}

void Tensor::set_cpu_data(const std::vector<float>& v) {
    if (data_.device != CPU)
        throw std::runtime_error("set_cpu_data: tensor is on a non-CPU device");
    std::copy(v.begin(), v.end(), data_.data());
}

void Tensor::zero_grad() {
    if (!grad_.valid()) return;
    get_ops(grad_.device).zero(grad_.data(), size());
}

// ---------------------------------------------------------------------------
// Graph wiring helper
// ---------------------------------------------------------------------------

void Tensor::_wire_grad(std::string op,
                        std::vector<std::shared_ptr<Tensor>> children,
                        std::function<void()> fwd,
                        std::function<void(const float*, int)> bwd)
{
    fwd();
    _op       = std::move(op);
    _children = std::move(children);
    _backward = std::move(bwd);
}

// ---------------------------------------------------------------------------
// Internal output factory
// ---------------------------------------------------------------------------

Tensor Tensor::_make_output() const {
    Tensor out;
    out.shape = shape;
    out.data_ = make_storage(data_.size, data_.device);
    out.grad_ = make_storage(data_.size, data_.device);
    get_ops(data_.device).zero(out.grad_.data(), size());
    return out;
}

Tensor Tensor::_make_output(std::vector<size_t> shape) const {
    int n = 1;
    for (size_t d : shape) n *= static_cast<int>(d);
    Tensor out;
    out.shape  = std::move(shape);
    out.data_  = make_storage(n, data_.device);
    out.grad_  = make_storage(n, data_.device);
    get_ops(data_.device).zero(out.grad_.data(), n);
    return out;
}

// ---------------------------------------------------------------------------
// Arithmetic operators
// ---------------------------------------------------------------------------

Tensor Tensor::operator+(const Tensor& other) const {
    check_same_shape(*this, other);
    check_same_device(*this, other);
    int n = size();
    Tensor out = _make_output();
    auto lhs = shared_self();
    auto rhs = shared_ref(other);
    out._wire_grad("+", {lhs, rhs},
        [&]() { get_ops(data_.device).add(out.data_.data(), data_.data(), other.data_.data(), n); },
        [lhs, rhs, n](const float* g, int) {
            const Ops& ops = get_ops(lhs->device());
            ops.axpy(n, 1.0f, g, lhs->grad_.data());
            ops.axpy(n, 1.0f, g, rhs->grad_.data());
        });
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_same_shape(*this, other);
    check_same_device(*this, other);
    int n = size();
    Tensor out = _make_output();
    auto lhs = shared_self();
    auto rhs = shared_ref(other);
    out._wire_grad("*", {lhs, rhs},
        [&]() { get_ops(data_.device).mul(out.data_.data(), data_.data(), other.data_.data(), n); },
        [lhs, rhs, n](const float* g, int) {
            const Ops& ops = get_ops(lhs->device());
            ops.mul_acc(lhs->grad_.data(), rhs->data_.data(), g, n);
            ops.mul_acc(rhs->grad_.data(), lhs->data_.data(), g, n);
        });
    return out;
}

Tensor Tensor::operator*(float scalar) const {
    int n = size();
    Tensor out = _make_output();
    auto self = shared_self();
    out._wire_grad("*scalar", {self},
        [&]() { get_ops(data_.device).scale(out.data_.data(), data_.data(), scalar, n); },
        [self, scalar, n](const float* g, int) {
            get_ops(self->device()).axpy(n, scalar, g, self->grad_.data());
        });
    return out;
}

Tensor Tensor::operator-() const {
    int n = size();
    Tensor out = _make_output();
    auto self = shared_self();
    out._wire_grad("neg", {self},
        [&]() { get_ops(data_.device).negate(out.data_.data(), data_.data(), n); },
        [self, n](const float* g, int) {
            get_ops(self->device()).axpy(n, -1.0f, g, self->grad_.data());
        });
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_same_shape(*this, other);
    check_same_device(*this, other);
    int n = size();
    Tensor out = _make_output();
    auto lhs = shared_self();
    auto rhs = shared_ref(other);
    out._wire_grad("-", {lhs, rhs},
        [&]() { get_ops(data_.device).sub(out.data_.data(), data_.data(), other.data_.data(), n); },
        [lhs, rhs, n](const float* g, int) {
            const Ops& ops = get_ops(lhs->device());
            ops.axpy(n,  1.0f, g, lhs->grad_.data());
            ops.axpy(n, -1.0f, g, rhs->grad_.data());
        });
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    check_same_shape(*this, other);
    check_same_device(*this, other);
    int n = size();
    Tensor out = _make_output();
    auto lhs = shared_self();
    auto rhs = shared_ref(other);
    out._wire_grad("/", {lhs, rhs},
        [&]() { get_ops(data_.device).div_op(out.data_.data(), data_.data(), other.data_.data(), n); },
        [lhs, rhs, n](const float* g, int) {
            const Ops& ops = get_ops(lhs->device());
            ops.div_acc_lhs(lhs->grad_.data(), g, rhs->data_.data(), n);
            ops.div_acc_rhs(rhs->grad_.data(), g, lhs->data_.data(), rhs->data_.data(), n);
        });
    return out;
}

Tensor Tensor::pow(float e) const {
    int n = size();
    Tensor out = _make_output();
    auto self = shared_self();
    out._wire_grad("**" + std::to_string(e), {self},
        [&]() { get_ops(data_.device).pow_op(out.data_.data(), data_.data(), e, n); },
        [self, e, n](const float* g, int) {
            get_ops(self->device()).pow_acc(self->grad_.data(), self->data_.data(), g, e, n);
        });
    return out;
}

// Matrix multiplication: (M, K) @ (K, N) -> (M, N)
Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2)
        throw std::runtime_error("matmul requires 2-D tensors");
    if (shape[1] != other.shape[0])
        throw std::runtime_error("matmul: inner dimensions must match");
    check_same_device(*this, other);

    int M = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);
    int N = static_cast<int>(other.shape[1]);

    Tensor out = _make_output({static_cast<size_t>(M), static_cast<size_t>(N)});
    auto lhs = shared_self();
    auto rhs = shared_ref(other);
    out._wire_grad("@", {lhs, rhs},
        // C = 1*A@B + 0*C, row-major: lda=K, ldb=N, ldc=N
        [&]() { get_ops(data_.device).gemm(false, false, M, N, K,
                    1.0f, data_.data(), K, other.data_.data(), N, 0.0f, out.data_.data(), N); },
        [lhs, rhs, M, K, N](const float* g, int) {
            const Ops& ops = get_ops(lhs->device());
            // dL/dA = g @ B^T - g:(M,N), B^T:(N,K) -> (M,K)
            ops.gemm(false, true, M, K, N,
                1.0f, g, N, rhs->data_.data(), N, 1.0f, lhs->grad_.data(), K);
            // dL/dB = A^T @ g - A^T:(K,M), g:(M,N) -> (K,N)
            ops.gemm(true, false, K, N, M,
                1.0f, lhs->data_.data(), K, g, N, 1.0f, rhs->grad_.data(), N);
        });
    return out;
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

// For activations that need forward output during backward, we capture
// the output Storage (a copy of the shared_ptr). This avoids a reference
// cycle (out_ptr capturing itself) and avoids any device->host copy.

Tensor Tensor::exp() const {
    int n = size();
    Tensor out = _make_output();
    auto self = shared_self();
    Storage out_storage = out.data_;
    out._wire_grad("exp", {self},
        [&]() { get_ops(data_.device).exp_op(out.data_.data(), data_.data(), n); },
        [self, out_storage, n](const float* g, int) {
            get_ops(self->device()).mul_acc(self->grad_.data(), out_storage.data(), g, n);
        });
    return out;
}

Tensor Tensor::log() const {
    int n = size();
    if (data_.device == CPU) {
        const float* d = data_.data();
        for (int i = 0; i < n; i++)
            if (d[i] <= 0.0f)
                throw std::domain_error("log requires all-positive Tensor");
    }
    Tensor out = _make_output();
    auto self = shared_self();
    out._wire_grad("log", {self},
        [&]() { get_ops(data_.device).log_op(out.data_.data(), data_.data(), n); },
        [self, n](const float* g, int) {
            get_ops(self->device()).log_acc(self->grad_.data(), self->data_.data(), g, n);
        });
    return out;
}

Tensor Tensor::relu() const {
    int n = size();
    Tensor out = _make_output();
    auto self = shared_self();
    Storage out_storage = out.data_;
    out._wire_grad("relu", {self},
        [&]() { get_ops(data_.device).relu_op(out.data_.data(), data_.data(), n); },
        [self, out_storage, n](const float* g, int) {
            get_ops(self->device()).relu_bwd(self->grad_.data(), out_storage.data(), g, n);
        });
    return out;
}

Tensor Tensor::tanh() const {
    int n = size();
    Tensor out = _make_output();
    auto self = shared_self();
    Storage out_storage = out.data_;
    out._wire_grad("tanh", {self},
        [&]() { get_ops(data_.device).tanh_op(out.data_.data(), data_.data(), n); },
        [self, out_storage, n](const float* g, int) {
            get_ops(self->device()).tanh_bwd(self->grad_.data(), out_storage.data(), g, n);
        });
    return out;
}

Tensor Tensor::sigmoid() const {
    int n = size();
    Tensor out = _make_output();
    auto self = shared_self();
    Storage out_storage = out.data_;
    out._wire_grad("sigmoid", {self},
        [&]() { get_ops(data_.device).sigmoid_op(out.data_.data(), data_.data(), n); },
        [self, out_storage, n](const float* g, int) {
            get_ops(self->device()).sigmoid_bwd(self->grad_.data(), out_storage.data(), g, n);
        });
    return out;
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

std::vector<Tensor*> Tensor::buildTopo() {
    std::vector<Tensor*> topo;
    std::unordered_set<const Tensor*> visited;
    std::function<void(Tensor*)> dfs = [&](Tensor* v) {
        if (!visited.contains(v)) {
            visited.insert(v);
            for (const auto& child : v->_children)
                dfs(child.get());
            topo.push_back(v);
        }
    };
    dfs(this);
    return topo;
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

void Tensor::backward(bool cache) {
    auto run = [this](std::vector<Tensor*>& topo) {
        get_ops(grad_.device).fill(grad_.data(), size(), 1.0f);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_backward((*it)->grad_.data(), (*it)->size());
    };

    if (cache && !_cachedTopo.empty()) {
        run(_cachedTopo);
    } else {
        auto topo = buildTopo();
        run(topo);
        if (cache) _cachedTopo = std::move(topo);
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

std::string Tensor::repr() const {
    std::ostringstream ss;
    auto d = cpu_data();
    auto g = cpu_grad();
    ss << "Tensor(data=[";
    for (size_t i = 0; i < d.size(); i++) {
        if (i) ss << ", ";
        ss << d[i];
    }
    ss << "], grad=[";
    for (size_t i = 0; i < g.size(); i++) {
        if (i) ss << ", ";
        ss << g[i];
    }
    ss << "], op='" << _op << "', label='" << _label << "')";
    return ss.str();
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    return stream << tensor.repr();
}
