#ifndef COMMON_HIP_H_
#define COMMON_HIP_H_

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define HIP_CHECK(Call)                                                        \
    do {                                                                       \
        hipError_t Error = (Call);                                             \
        if (Error != hipSuccess) {                                             \
            fprintf(stderr, "HIP Error at %s:%d: %s (%d)\n",                   \
                    __FILE__, __LINE__, hipGetErrorString(Error), Error);      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#endif // COMMON_HIP_H_
