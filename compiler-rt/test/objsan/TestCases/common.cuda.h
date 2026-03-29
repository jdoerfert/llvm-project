#ifndef COMMON_CUDA_H_
#define COMMON_CUDA_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(Call)                                                       \
    do {                                                                       \
        cudaError_t Error = (Call);                                            \
        if (Error != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error at %s:%d: %s (%d)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(Error), Error);     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#endif // COMMON_CUDA_H_
