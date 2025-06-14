#include "xlpd.h"
#include <iostream>

void check(cudaError_t err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::cerr << func << " " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
