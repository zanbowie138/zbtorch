#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <zbtorch_ext/tensor.h>

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
        .def("__mul__",      &Tensor::operator*)
        .def("__rmul__",     [](std::shared_ptr<Tensor> t, float o) {
                                 return *std::make_shared<Tensor>(o) * *t; })
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
        // Backprop
        .def("backward", &Tensor::backward)
        // Repr
        .def("__repr__", &Tensor::repr);
}
