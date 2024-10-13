// RUN: %libomptarget-compileopt-generic -fsanitize=offload \
// RUN:   -fopenmp-offload-mandatory
// RUN: %libomptarget-run-generic

// REQUIRES: gpu

#include <omp.h>

#define __SHARED__ __attribute__((address_space(3)))
#define EL_TYPE int

EL_TYPE global_arr[3] = {1, 2, 3};
#pragma omp declare target(global_arr)

__SHARED__ EL_TYPE shared_arr [[clang::loader_uninitialized]][3];
#pragma omp declare target to(shared_arr) device_type(nohost)

#pragma omp declare target
void test_ptr(EL_TYPE *unknown_ptr, int index) { unknown_ptr[index] = 222932; }
#pragma omp end declare target

int main() {
#pragma omp target
  {
    test_ptr((EL_TYPE *)global_arr, 1);
    test_ptr((EL_TYPE *)shared_arr, 2);
    test_ptr((EL_TYPE *)(shared_arr + 1), 1);

    EL_TYPE *unknown_arr = (EL_TYPE *)shared_arr;
    test_ptr(unknown_arr, 2);

    int offset = 1;
    EL_TYPE *unknown_arr_off = (EL_TYPE *)shared_arr + offset;
    test_ptr(unknown_arr_off, 1);
  }
  return 0;
}
