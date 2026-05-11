#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_CHECK(Call)                                                        \
    do {                                                                       \
        hipError_t Error = (Call);                                             \
        if (Error != hipSuccess) {                                             \
            fprintf(stderr, "HIP Error at %s:%d: %s (%d)\n",                   \
                    __FILE__, __LINE__, hipGetErrorString(Error), Error);      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

extern "C" {
void objsan_rt_init(void);
void objsan_rt_deinit(void);
}

__device__ void func(int *array, int size, int n) {
  array[n] = 200; // NOTE: Should trigger an error
}

__global__ void kernel(int *array, int size, int n) {
  func(array, size, n);
}

int main(int argc, char **argv) {
  objsan_rt_init();

  const int size = 10;
  const int n = (argc == 1) ? 10 : 9;

  int *d_array;
  HIP_CHECK(hipMalloc((void **)&d_array, size * sizeof(int)));

  kernel<<<1, 1>>>(d_array, size, n);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(d_array));

  fprintf(stdout, "%s", "Execution completed successfully\n");

  objsan_rt_deinit();
}
