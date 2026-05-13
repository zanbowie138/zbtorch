#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "device.h"
#include "storage.h"
#include "ops.h"

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Storage data_;
    Storage grad_;
    std::vector<size_t> shape;
    std::string _op;
    std::string _label;
    std::vector<std::shared_ptr<Tensor>> _children;
    std::vector<Tensor*> _cachedTopo;

    // Receives this tensor's raw device gradient pointer during backward.
    std::function<void(const float*, int)> _backward;

    // Constructors
    Tensor();
    explicit Tensor(float scalar, Device device = CPU, std::string label = "");
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape,
           Device device = CPU, std::string label = "");

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    // Device and size queries
    Device device() const { return data_.device; }
    int size() const { return static_cast<int>(data_.size); }

    // Returns a new Tensor with data copied to device d. Does not copy the graph.
    Tensor to(Device d) const;

    // Host-side accessors — for Python bindings and debugging.
    // On CUDA these would do a device→host transfer (not yet implemented).
    std::vector<float> data() const;
    std::vector<float> grad() const;
    void set_data(const std::vector<float>& v);
    void zero_grad();

    // Arithmetic operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    Tensor operator-() const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor pow(float e) const;
    Tensor matmul(const Tensor& other) const;

    // Activation functions
    Tensor exp() const;
    Tensor log() const;
    Tensor relu() const;
    Tensor tanh() const;
    Tensor sigmoid() const;

    std::vector<Tensor*> buildTopo();
    void backward(bool cache = true);

    std::string repr() const;
    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

private:
    // Allocates an output Tensor with the same device and size as *this.
    Tensor _make_output() const;
    // Allocates an output Tensor with an arbitrary shape on the same device.
    Tensor _make_output(std::vector<size_t> shape) const;

    std::shared_ptr<Tensor> shared_self() const {
        return const_cast<Tensor*>(this)->shared_from_this();
    }
    static std::shared_ptr<Tensor> shared_ref(const Tensor& t) {
        return const_cast<Tensor*>(&t)->shared_from_this();
    }

    // Runs the forward kernel then sets _op, _children, and _backward.
    void _wire_grad(std::string op,
                    std::vector<std::shared_ptr<Tensor>> children,
                    std::function<void()> fwd,
                    std::function<void(const float*, int)> bwd);
};

template <typename... Args>
std::shared_ptr<Tensor> make_tensor(Args&&... args) {
    return std::make_shared<Tensor>(std::forward<Args>(args)...);
}

// ---------------------------------------------------------------------------
// Operators on shared_ptr<Tensor>
// ---------------------------------------------------------------------------

inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return std::make_shared<Tensor>(*a + *b);
}
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return std::make_shared<Tensor>(*a * *b);
}
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, float s) {
    return std::make_shared<Tensor>(*a * s);
}
inline std::shared_ptr<Tensor> operator*(float s, const std::shared_ptr<Tensor>& a) {
    return std::make_shared<Tensor>(*a * s);
}
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a) {
    return std::make_shared<Tensor>(-*a);
}
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return std::make_shared<Tensor>(*a - *b);
}
inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return std::make_shared<Tensor>(*a / *b);
}
inline std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return std::make_shared<Tensor>(a->matmul(*b));
}
