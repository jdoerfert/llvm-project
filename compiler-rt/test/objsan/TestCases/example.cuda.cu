// RUN: %clang_objsan_cuda_compile -DCONF1 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF1
// CONF1: s bad

// RUN: %clang_objsan_cuda_compile -DCONF2 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF2
// CONF2: Execution completed successfully

// RUN: %clang_objsan_cuda_compile -DCONF3 -c %s -o %t.o
// RUN: %clang_objsan_cuda_link %t.o %clang_objsan_cuda_post_link -o %t.a.out
// RUN: not %t.a.out 2>&1 | FileCheck %s --check-prefix=CONF1
// CONF3: s bad

#include <cstdlib>
#include <cuda_runtime.h>
#include <cstdio>

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


__device__ void func(int *array, int size) {
#ifdef CONF1
  array[10] = 0;
#endif
#ifdef CONF2
  array[9] = 0;
#endif
#ifdef CONF3
  array[-1] = 0;
#endif
}

__global__ void kernel(int *array, int size) {
	//printf("kernel: array[%d] %d\n", 0, array[0]);
	//printf("kernel: array %p %d\n", array, size);
	func(array, size);
	//printf("kernel: array %p size %d\n", array, size);
	//printf("kernel: array[%d] %d\n", 0, array[0]);
}

int main(int argc, char **argv) {
	const int size = 10;
	int *d_array;

	CUDA_CHECK(cudaMalloc((void**)&d_array, size * sizeof(int)));

	kernel<<<1, 1>>>(d_array, size);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaFree(d_array));

    fprintf(stdout, "%s", "Execution completed successfully\n");
}
