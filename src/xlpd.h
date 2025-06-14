#ifndef XLPD_H
#define XLPD_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define CHECK(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file, const int line);

namespace py = pybind11;
py::array_t<uint8_t> color_to_grayscale(py::array_t<uint8_t> pin);
py::array_t<float> vector_add(py::array_t<float> a, py::array_t<float> b);
py::array_t<uint8_t> blur(py::array_t<uint8_t> pin);

#endif