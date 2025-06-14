#include "xlpd.h"
#include <iostream>

__global__ void vector_add_kernel(float *a, float *b, float *c, size_t n)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

py::array_t<float> vector_add(py::array_t<float> a, py::array_t<float> b)
{
    // Request buffers
    py::buffer_info a_buf = a.request();
    py::buffer_info b_buf = b.request();

    if (a_buf.size != b_buf.size)
    {
        throw std::runtime_error("Input arrays must have the same size.");
    }

    auto result = py::array_t<float>(a_buf.size);
    py::buffer_info r_buf = result.request();

    // Get pointers
    float *a_h = static_cast<float *>(a_buf.ptr);
    float *b_h = static_cast<float *>(b_buf.ptr);
    float *r_h = static_cast<float *>(r_buf.ptr);

    size_t n = a_buf.size;
    size_t size = a_buf.size * sizeof(float);
    std::cout << "size: " << size << std::endl;
    float *a_d, *b_d, *r_d;
    CHECK(cudaMalloc((void **)&a_d, size));
    CHECK(cudaMalloc((void **)&b_d, size));
    CHECK(cudaMalloc((void **)&r_d, size));

    CHECK(cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice));

    vector_add_kernel<<<ceil(n / 256.0f), 256>>>(a_d, b_d, r_d, n);

    CHECK(cudaMemcpy(r_h, r_d, size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(a_d));
    CHECK(cudaFree(b_d));
    CHECK(cudaFree(r_d));

    return result;
}
