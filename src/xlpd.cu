#include "xlpd.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(_core, m)
{
    m.def("vector_add", &vector_add, "vector_add");
    m.def("color_to_grayscale", &color_to_grayscale, "color_to_grayscale");
    m.def("blur", &blur, "blur");
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
}
