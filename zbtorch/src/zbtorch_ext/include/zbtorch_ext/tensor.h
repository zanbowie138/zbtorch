#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <ostream>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<size_t> shape;
    std::string _op;
    std::string _label;
    std::vector<std::shared_ptr<Tensor>> _children;
    std::vector<Tensor*> _cachedTopo;

    // _backward receives this tensor's grad and propagates it to _children
    std::function<void(const std::vector<float>&)> _backward;

    // Constructors
    Tensor();
    explicit Tensor(float scalar, std::string label = "");
    Tensor(std::vector<float> data, std::vector<size_t> shape, std::string label = "");

    // Rule of five
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    // Arithmetic operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    Tensor operator-() const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor pow(float exp) const;
    Tensor matmul(const Tensor& other) const;

    // Activation functions
    Tensor exp() const;
    Tensor log() const;
    Tensor relu() const;
    Tensor tanh() const;
    Tensor sigmoid() const;

    std::vector<Tensor *> buildTopo();

    // Backpropogate and update gradients of self and children
    void backward(bool cache = true);

    std::string repr() const;

    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

private:
    Tensor _make_output() const;
};
