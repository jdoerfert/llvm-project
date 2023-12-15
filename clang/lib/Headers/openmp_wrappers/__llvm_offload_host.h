#include "__llvm_offload.h"

extern "C" {
unsigned llvmLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                          void **args, size_t sharedMem = 0, void *stream = 0);
}
