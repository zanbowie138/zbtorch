#include <format>
#include <ranges>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <core/tensor.h>
#include <core/neuron.h>
#include <core/device.h>
#include <bindings/device_str.h>
#include <cuda_runtime.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m, py::mod_gil_not_used()) {
    m.doc() = "zbtorch C++ extension";

    py::class_<Device>(m, "device")
        .def(py::init([](const py::object &obj) { return parseDevice(obj); }), py::arg("type"))
        .def_property_readonly("type", [](Device d) { return getDeviceName(d); })
        .def("__str__",  [](const Device d) { return getDeviceName(d); })
        .def("__repr__", [](const Device d) { return std::format("device('{}')", getDeviceName(d)); })
        .def("__eq__",   [](const Device d, const py::object& other) -> py::object {
            if (py::isinstance<Device>(other))
                return py::cast(d == other.cast<Device>());
            if (py::isinstance<py::str>(other))
                return py::cast(getDeviceName(d) == other.cast<std::string_view>());
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        })
        .def("__hash__", [](const Device d) { return py::hash(py::str(getDeviceName(d))); });

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        // Constructors
        // scalar initializer
        .def(py::init([](float scalar, const py::object &device, const std::string& label) {
                return std::make_unique<Tensor>(scalar, parseDevice(device), label);
             }), py::arg("scalar"), py::arg("device") = py::str("cpu"), py::arg("label") = "")
        // numpy initializer
        .def(py::init([](const py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                         const py::object& device, const std::string& label) {
                auto buf = arr.request();
                std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
                std::vector<float> data(static_cast<float*>(buf.ptr),
                                        static_cast<float*>(buf.ptr) + buf.size);
                return std::make_unique<Tensor>(data, shape, parseDevice(device), label);
             }), py::arg("data"), py::arg("device") = py::str("cpu"), py::arg("label") = "")
        // manual (vector, shape) initializer
        .def(py::init([](const std::vector<float>& data, const std::vector<size_t>& shape,
                         const py::object &device, const std::string& label) {
                return std::make_unique<Tensor>(data, shape, parseDevice(device), label);
             }), py::arg("data"), py::arg("shape"), py::arg("device") = py::str("cpu"), py::arg("label") = "")

        .def_property("data",
            [](const Tensor& t) {
                auto v = t.data();
                return py::array_t<float>({static_cast<py::ssize_t>(v.size())}, v.data());
            },
            [](Tensor& t, const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
                auto buf = arr.request();
                std::vector<float> v(static_cast<float*>(buf.ptr),
                                     static_cast<float*>(buf.ptr) + buf.size);
                t.set_data(v);
            })
        .def_property_readonly("grad",
            [](const Tensor& t) {
                auto v = t.grad();
                return py::array_t<float>({static_cast<py::ssize_t>(v.size())}, v.data());
            })
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("_op", &Tensor::_op)
        .def_readwrite("_label", &Tensor::_label)

        // Expose _children as a Python set of shared_ptr<Tensor>
        .def_property("_children",
            [](const Tensor& t) -> py::set {
                py::set result;
                for (const auto& child : t._children)
                    result.add(py::cast(child));
                return result;
            },
            [](Tensor& t, const py::set& children) {
                t._children.clear();
                for (auto item : children)
                    t._children.push_back(item.cast<std::shared_ptr<Tensor>>());
            })
        // Arithmetic
        .def("__add__",      &Tensor::operator+)
        .def("__radd__",     [](const std::shared_ptr<Tensor>& t, float o) {
                                 return *std::make_shared<Tensor>(o, t->device()) + *t; })
        .def("__mul__",      py::overload_cast<const Tensor&>(&Tensor::operator*, py::const_))
        .def("__mul__",      py::overload_cast<float>(&Tensor::operator*, py::const_))
        .def("__rmul__",     [](const Tensor& t, float o) { return t * o; })
        .def("__neg__",      [](const Tensor& t) { return -t; })
        .def("__sub__",      py::overload_cast<const Tensor&>(&Tensor::operator-, py::const_))
        .def("__rsub__",     [](const std::shared_ptr<Tensor>& t, float o) {
                                 auto neg = std::make_shared<Tensor>(-*t);
                                 return *std::make_shared<Tensor>(o, t->device()) + *neg; })
        .def("__truediv__",  &Tensor::operator/)
        .def("__rtruediv__", [](const std::shared_ptr<Tensor>& t, float o) {
                                 auto inv = std::make_shared<Tensor>(t->pow(-1.0f));
                                 return *std::make_shared<Tensor>(o, t->device()) * *inv; })
        .def("__pow__",      &Tensor::pow)
        .def("__matmul__",   &Tensor::matmul)
        .def("matmul",       &Tensor::matmul)
        // Activations
        .def("exp",     &Tensor::exp)
        .def("log",     &Tensor::log)
        .def("relu",    &Tensor::relu)
        .def("tanh",    &Tensor::tanh)
        .def("sigmoid", &Tensor::sigmoid)
        .def_property_readonly("device", [](const Tensor& t) { return getDeviceName(t.device()); })
        .def("to", [](const Tensor& t, const py::object& d) { return t.to(parseDevice(d)); },
             py::arg("device"))
        .def("zero_grad", &Tensor::zero_grad)
        // Topology
        .def("build_topo", [](Tensor& t) -> py::set {
            py::set result;
            for (const auto& child : t.buildTopo())
                result.add(py::cast(child));
            return result;
        }, "Returns a list of the topology of this Tensor and it's children.")
        // Backprop
        .def("backward", &Tensor::backward, py::arg("cache") = true)
        // Repr
        .def("__repr__", &Tensor::repr);

    py::class_<Neuron>(m, "Neuron")
        .def(py::init<int>(), py::arg("n_inputs"))
        .def("__call__",   &Neuron::forward,    py::arg("x"))
        .def("forward",    &Neuron::forward,    py::arg("x"))
        .def("parameters", &Neuron::parameters)
        .def("zero_grad",  &Neuron::zero_grad)
        .def_readwrite("w", &Neuron::w)
        .def_readwrite("b", &Neuron::b);

    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int>(), py::arg("n_inputs"), py::arg("n_outputs"))
        .def("__call__",   &Layer::forward,     py::arg("x"))
        .def("forward",    &Layer::forward,     py::arg("x"))
        .def("parameters", &Layer::parameters)
        .def("zero_grad",  &Layer::zero_grad)
        .def_readwrite("neurons", &Layer::neurons);

    py::class_<MLP>(m, "MLP")
        .def(py::init<int, std::vector<int>>(),
             py::arg("n_inputs"), py::arg("layer_sizes"))
        .def("__call__",   &MLP::forward,       py::arg("x"))
        .def("forward",    &MLP::forward,       py::arg("x"))
        .def("parameters", &MLP::parameters)
        .def("zero_grad",  &MLP::zero_grad)
        .def_readwrite("layers", &MLP::layers);

    m.def("_cuda_is_available", []() {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count > 0;
    });
}
