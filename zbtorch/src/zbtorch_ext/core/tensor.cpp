#include <cblas.h>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <zbtorch_ext/tensor.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape)
        throw std::runtime_error("Tensor shape mismatch");
}

Tensor Tensor::_make_output() const {
    Tensor out;
    out.shape = shape;
    out.data.resize(data.size());
    out.grad.assign(data.size(), 0.0f);
    return out;
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

Tensor::Tensor()
    : _backward([](const std::vector<float>&) {}) {}

Tensor::Tensor(float scalar, std::string label)
    : data({scalar}), grad({0.0f}), shape({1}), _label(std::move(label)),
      _backward([](const std::vector<float>&) {}) {}

Tensor::Tensor(std::vector<float> d, std::vector<size_t> s, std::string label)
    : data(std::move(d)), shape(std::move(s)), _label(std::move(label)),
      _backward([](const std::vector<float>&) {}) {
    grad.assign(data.size(), 0.0f);
}

// ---------------------------------------------------------------------------
// Rule of five
// ---------------------------------------------------------------------------

Tensor::Tensor(const Tensor& other)
    : data(other.data), grad(other.grad), shape(other.shape),
      _op(other._op), _label(other._label),
      _children(other._children), _backward(other._backward) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    data = other.data;
    grad = other.grad;
    shape = other.shape;
    _op = other._op;
    _label = other._label;
    _children = other._children;
    _backward = other._backward;
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(std::move(other.data)), grad(std::move(other.grad)),
      shape(std::move(other.shape)), _op(std::move(other._op)),
      _label(std::move(other._label)), _children(std::move(other._children)),
      _backward(std::move(other._backward)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    data = std::move(other.data);
    grad = std::move(other.grad);
    shape = std::move(other.shape);
    _op = std::move(other._op);
    _label = std::move(other._label);
    _children = std::move(other._children);
    _backward = std::move(other._backward);
    return *this;
}

Tensor::~Tensor() = default;

// ---------------------------------------------------------------------------
// Arithmetic operators
// ---------------------------------------------------------------------------

Tensor Tensor::operator+(const Tensor& other) const {
    check_same_shape(*this, other);
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = data[i] + other.data[i];
    out._op = "+";
    auto lhs = const_cast<Tensor*>(this)->shared_from_this();
    auto rhs = const_cast<Tensor*>(&other)->shared_from_this();
    out._children = {lhs, rhs};
    out._backward = [lhs, rhs, n](const std::vector<float>& g) {
        cblas_saxpy(n,  1.0f, g.data(), 1, lhs->grad.data(), 1);
        cblas_saxpy(n,  1.0f, g.data(), 1, rhs->grad.data(), 1);
    };
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_same_shape(*this, other);
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = data[i] * other.data[i];
    out._op = "*";
    auto lhs = const_cast<Tensor*>(this)->shared_from_this();
    auto rhs = const_cast<Tensor*>(&other)->shared_from_this();
    out._children = {lhs, rhs};
    out._backward = [lhs, rhs, n](const std::vector<float>& g) {
        for (int i = 0; i < n; i++) lhs->grad[i] += rhs->data[i] * g[i];
        for (int i = 0; i < n; i++) rhs->grad[i] += lhs->data[i] * g[i];
    };
    return out;
}

Tensor Tensor::operator*(float scalar) const {
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    cblas_scopy(n, data.data(), 1, out.data.data(), 1);
    cblas_sscal(n, scalar, out.data.data(), 1);
    out._op = "*scalar";
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    out._children = {self};
    out._backward = [self, scalar, n](const std::vector<float>& g) {
        cblas_saxpy(n, scalar, g.data(), 1, self->grad.data(), 1);
    };
    return out;
}

Tensor Tensor::operator-() const {
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = -data[i];
    out._op = "neg";
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    out._children = {self};
    out._backward = [self, n](const std::vector<float>& g) {
        cblas_saxpy(n, -1.0f, g.data(), 1, self->grad.data(), 1);
    };
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_same_shape(*this, other);
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = data[i] - other.data[i];
    out._op = "-";
    auto lhs = const_cast<Tensor*>(this)->shared_from_this();
    auto rhs = const_cast<Tensor*>(&other)->shared_from_this();
    out._children = {lhs, rhs};
    out._backward = [lhs, rhs, n](const std::vector<float>& g) {
        cblas_saxpy(n,  1.0f, g.data(), 1, lhs->grad.data(), 1);
        cblas_saxpy(n, -1.0f, g.data(), 1, rhs->grad.data(), 1);
    };
    return out;
}

// Elementwise division
Tensor Tensor::operator/(const Tensor& other) const {
    check_same_shape(*this, other);
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = data[i] / other.data[i];
    out._op = "/";
    auto lhs = const_cast<Tensor*>(this)->shared_from_this();
    auto rhs = const_cast<Tensor*>(&other)->shared_from_this();
    out._children = {lhs, rhs};
    out._backward = [lhs, rhs, n](const std::vector<float>& g) {
        for (int i = 0; i < n; i++) lhs->grad[i] += g[i] / rhs->data[i];
        for (int i = 0; i < n; i++) rhs->grad[i] += -lhs->data[i] / (rhs->data[i] * rhs->data[i]) * g[i];
    };
    return out;
}

Tensor Tensor::pow(float e) const {
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = std::pow(data[i], e);
    out._op = "**" + std::to_string(e);
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    out._children = {self};
    out._backward = [self, e, n](const std::vector<float>& g) {
        for (int i = 0; i < n; i++)
            self->grad[i] += e * std::pow(self->data[i], e - 1.0f) * g[i];
    };
    return out;
}

// Matrix multiplication: (M, K) @ (K, N) -> (M, N)
Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2)
        throw std::runtime_error("matmul requires 2-D tensors");
    if (shape[1] != other.shape[0])
        throw std::runtime_error("matmul shape mismatch: inner dims must match");

    size_t M = shape[0];
    size_t N = other.shape[1];

    int M_i = static_cast<int>(shape[0]);
    int K_i = static_cast<int>(shape[1]);
    int N_i = static_cast<int>(other.shape[1]);

    Tensor out;
    out.shape = { M, N };
    out.data.resize(M * N);
    out.grad.assign(M * N, 0.0f);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M_i, N_i, K_i,
                1.0f, data.data(), K_i,
                       other.data.data(), N_i,
                0.0f, out.data.data(), N_i);

    out._op = "@";
    auto lhs = const_cast<Tensor*>(this)->shared_from_this();
    auto rhs = const_cast<Tensor*>(&other)->shared_from_this();
    out._children = {lhs, rhs};

    out._backward = [lhs, rhs, M_i, K_i, N_i](const std::vector<float>& g) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M_i, K_i, N_i,
                    1.0f, g.data(), N_i,
                           rhs->data.data(), N_i,
                    1.0f, lhs->grad.data(), K_i);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K_i, N_i, M_i,
                    1.0f, lhs->data.data(), K_i,
                           g.data(), N_i,
                    1.0f, rhs->grad.data(), N_i);
    };

    return out;
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

Tensor Tensor::exp() const {
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = std::exp(data[i]);
    out._op = "exp";
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    std::vector<float> out_vals = out.data;
    out._children = {self};
    out._backward = [self, out_vals](const std::vector<float>& g) {
        for (size_t i = 0; i < g.size(); i++) self->grad[i] += out_vals[i] * g[i];
    };
    return out;
}

Tensor Tensor::log() const {
    for (float v : data)
        if (v <= 0.0f)
            throw std::domain_error("log is only defined for positive Tensors");
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = std::log(data[i]);
    out._op = "log";
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    out._children = {self};
    out._backward = [self, n](const std::vector<float>& g) {
        for (int i = 0; i < n; i++) self->grad[i] += g[i] / self->data[i];
    };
    return out;
}

Tensor Tensor::relu() const {
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    out._op = "relu";
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    std::vector<float> mask(n);
    for (int i = 0; i < n; i++) mask[i] = data[i] > 0.0f ? 1.0f : 0.0f;
    out._children = {self};
    out._backward = [self, mask](const std::vector<float>& g) {
        for (size_t i = 0; i < g.size(); i++) self->grad[i] += mask[i] * g[i];
    };
    return out;
}

Tensor Tensor::tanh() const {
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    for (int i = 0; i < n; i++)
        out.data[i] = std::tanh(data[i]);
    out._op = "tanh";
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    std::vector<float> dtanh(n);
    for (int i = 0; i < n; i++) dtanh[i] = 1.0f - out.data[i] * out.data[i];
    out._children = {self};
    out._backward = [self, dtanh](const std::vector<float>& g) {
        for (size_t i = 0; i < g.size(); i++) self->grad[i] += dtanh[i] * g[i];
    };
    return out;
}

Tensor Tensor::sigmoid() const {
    int n = static_cast<int>(data.size());
    Tensor out = _make_output();
    std::vector<float> s(n);
    for (int i = 0; i < n; i++) {
        s[i] = 1.0f / (1.0f + std::exp(-data[i]));
        out.data[i] = s[i];
    }
    out._op = "sigmoid";
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    std::vector<float> ds(n);
    for (int i = 0; i < n; i++) ds[i] = s[i] * (1.0f - s[i]);
    out._children = {self};
    out._backward = [self, ds](const std::vector<float>& g) {
        for (size_t i = 0; i < g.size(); i++) self->grad[i] += ds[i] * g[i];
    };
    return out;
}

std::vector<Tensor *> Tensor::buildTopo() {
    std::vector<Tensor*> topo;
    std::unordered_set<const Tensor*> visited;
    std::function<void(Tensor*)> build_topo = [&](Tensor* v) {
        if (!visited.count(v)) {
            visited.insert(v);
            for (const auto& child : v->_children)
                build_topo(child.get());
            topo.push_back(v);
        }
    };
    build_topo(this);
    return topo;
}

// ---------------------------------------------------------------------------
// Backprop
// ---------------------------------------------------------------------------

void Tensor::backward(bool cache) {
    if (cache && !_cachedTopo.empty()) {
        grad.assign(data.size(), 1.0f);

        // Iterate backwards through node topology
        for (auto it = _cachedTopo.rbegin(); it != _cachedTopo.rend(); ++it)
            (*it)->_backward((*it)->grad);
    }
    else {
        // Build topology
        auto topo = buildTopo();
        grad.assign(data.size(), 1.0f);

        // Iterate backwards through topology
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_backward((*it)->grad);

        // Save topology for future calls (if cached flag)
        if (cache) {
            _cachedTopo = std::move(topo);
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

std::string Tensor::repr() const {
    std::ostringstream ss;
    ss << "Tensor(data=[";
    for (size_t i = 0; i < data.size(); i++) {
        if (i) ss << ", ";
        ss << data[i];
    }
    ss << "], grad=[";
    for (size_t i = 0; i < grad.size(); i++) {
        if (i) ss << ", ";
        ss << grad[i];
    }
    ss << "], op='" << _op << "', label='" << _label << "')";
    return ss.str();
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    return stream << tensor.repr();
}
