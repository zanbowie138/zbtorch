#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <zbtorch_ext/tensor.h>
#include <zbtorch_ext/neuron.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m, py::mod_gil_not_used()) {
    m.doc() = "zbtorch C++ extension";

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<float, std::string>(),
             py::arg("scalar"), py::arg("label") = "")
        .def(py::init<std::vector<float>, std::vector<size_t>, std::string>(),
             py::arg("data"), py::arg("shape"), py::arg("label") = "")
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
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
            [](Tensor& t, py::set children) {
                t._children.clear();
                for (auto item : children)
                    t._children.push_back(item.cast<std::shared_ptr<Tensor>>());
            })
        // Arithmetic
        .def("__add__",      &Tensor::operator+)
        .def("__radd__",     [](std::shared_ptr<Tensor> t, float o) {
                                 return *std::make_shared<Tensor>(o) + *t; })
        .def("__mul__",      py::overload_cast<const Tensor&>(&Tensor::operator*, py::const_))
        .def("__mul__",      py::overload_cast<float>(&Tensor::operator*, py::const_))
        .def("__rmul__",     [](const Tensor& t, float o) { return t * o; })
        .def("__neg__",      [](const Tensor& t) { return -t; })
        .def("__sub__",      py::overload_cast<const Tensor&>(&Tensor::operator-, py::const_))
        .def("__rsub__",     [](std::shared_ptr<Tensor> t, float o) {
                                 return *std::make_shared<Tensor>(o) + (-*t); })
        .def("__truediv__",  &Tensor::operator/)
        .def("__rtruediv__", [](std::shared_ptr<Tensor> t, float o) {
                                 return *std::make_shared<Tensor>(o) * t->pow(-1.0f); })
        .def("__pow__",      &Tensor::pow)
        .def("__matmul__",   &Tensor::matmul)
        .def("matmul",       &Tensor::matmul)
        // Activations
        .def("exp",     &Tensor::exp)
        .def("log",     &Tensor::log)
        .def("relu",    &Tensor::relu)
        .def("tanh",    &Tensor::tanh)
        .def("sigmoid", &Tensor::sigmoid)
        // Topology
        .def("build_topo",    &Tensor::buildTopo, "Returns a list of the topology of this Tensor and it's children.")
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
}
