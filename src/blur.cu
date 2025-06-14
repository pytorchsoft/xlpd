#include "xlpd.h"

#define BLUR_SIZE 2

__global__ void blur_kernel(unsigned char *in, unsigned char *out, int width, int height)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= width || row >= height)
        return;

    int pixVal = 0;
    int pixels = 0;

    // Computer the average of the neighboring pixels
    for (int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE + 1; ++blurrow)
    {
        for (int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE + 1; ++blurcol)
        {
            int currow = row + blurrow;
            int curcol = col + blurcol;

            // Check if the current pixel is in the image
            if (curcol < 0 || curcol >= width || currow < 0 || currow >= height)
                continue;

            pixVal += in[currow * width + curcol];
            ++pixels; // Count the number of pixel values that have been added
        }
    }
    // printf("%d %d\n", pixVal, pixels);
    // Write out the result for this pixel
    out[row * width + col] = (unsigned char)((float)pixVal / pixels);
}

py::array_t<uint8_t> blur(py::array_t<uint8_t> pin)
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
    blur_kernel<<<grid, block>>>(pin_d, pout_d, width, height);
    CHECK(cudaMemcpy(pout_h, pout_d, pout.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(pin_d));
    CHECK(cudaFree(pout_d));
    CHECK(cudaDeviceSynchronize());
    return pout;
}
