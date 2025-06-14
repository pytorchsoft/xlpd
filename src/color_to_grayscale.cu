#include "xlpd.h"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

__global__ void color_to_grayscale_kernel(uint8_t *pin, uint8_t *pout, ssize_t width, ssize_t height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        int gray = row * width + col;
        int rgb = gray * 3;
        uint8_t r = pin[rgb];
        uint8_t g = pin[rgb + 1];
        uint8_t b = pin[rgb + 2];
        pout[gray] = 0.21 * r + 0.71 * g + 0.07 * b;
    }
}

py::array_t<uint8_t> color_to_grayscale(py::array_t<uint8_t> pin)
{
    ssize_t height = pin.shape(0);
    ssize_t width = pin.shape(1);
    auto pout = py::array_t<uint8_t>({height, width});
    uint8_t *pin_h = pin.mutable_data();
    uint8_t *pout_h = pout.mutable_data();
    uint8_t *pin_d;
    uint8_t *pout_d;

    CHECK(cudaMalloc((void **)&pin_d, pin.size()));
    CHECK(cudaMalloc((void **)&pout_d, pout.size()));
    CHECK(cudaMemcpy(pin_d, pin_h, pin.size(), cudaMemcpyHostToDevice));

    unsigned int w = ceil(width / 16.0f);
    unsigned int h = ceil(height / 16.0f);
    dim3 block = {16, 16, 1};
    dim3 grid = {w, h, 1};
    color_to_grayscale_kernel<<<grid, block>>>(pin_d, pout_d, width, height);
    CHECK(cudaMemcpy(pout_h, pout_d, pout.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(pin_d));
    CHECK(cudaFree(pout_d));
    CHECK(cudaDeviceSynchronize());
    return pout;
}
