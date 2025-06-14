cmake_minimum_required(VERSION 3.28)

project(xlpd LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(SRCS
    src/check.cu
    src/vector_add.cu
    src/color_to_grayscale.cu
    src/xlpd.cu
)

pybind11_add_module(_core ${SRCS})

target_compile_definitions(_core PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})

install(TARGETS _core DESTINATION xlpd)