cmake_minimum_required(VERSION 3.28)

project(xlpd CXX CUDA)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(_core MODULE src/main.cu WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)

target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)


install(TARGETS _core DESTINATION xlpd)