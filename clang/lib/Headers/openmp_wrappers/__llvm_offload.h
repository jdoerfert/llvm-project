#include <stdlib.h>

#define __global__ __attribute__((global))

extern "C" {

typedef struct dim3 {
  dim3() {}
  dim3(unsigned x) : x(x) {}
  unsigned x = 0, y = 0, z = 0;
} dim3;

// TODO: For some reason the CUDA device compilation requires this declaration
// to be present but it should not.
unsigned __llvmPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem = 0, void *stream = 0);
}
