#ifndef COMMON_H_
#define COMMON_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(Ret)                                                        \
  do {                                                                         \
    gpuAssert((Ret), __FILE__, __LINE__);                                      \
  } while (0)
namespace {
void gpuAssert(cudaError_t Ret, const char *File, int Line) {
  if (Ret != cudaSuccess) {
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(Ret), File,
            Line);
    exit(Ret);
  }
}
} // namespace

#endif // COMMON_H_
