#include "common.cuda.h"

__device__ int global2[20];

static __attribute__((noinline)) __device__ void get(int *array) {
#ifdef CONF1
  array[20] = 0;
#endif
#ifdef CONF2
  array[19] = 0;
#endif
}

static __global__ void kernel(int *array, int size) {
  get(global2);
  __syncthreads();
  array[0] = global2[0];
}

void call_kernel2(int *d_array, int size) {
  kernel<<<1, 1>>>(d_array, size);
}
