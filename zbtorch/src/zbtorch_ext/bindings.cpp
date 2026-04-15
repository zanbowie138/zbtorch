#include <pybind11/pybind11.h>
#include <zbtorch_ext/tensor.h>
namespace py = pybind11;

PYBIND11_MODULE(_C, m, py::mod_gil_not_used()) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}