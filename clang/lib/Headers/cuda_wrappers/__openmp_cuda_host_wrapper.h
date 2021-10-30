/*===---- __openmp_cuda_host_wrapper.h - CUDA host support for OpenMP ------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __OPENMP_CUDA_HOST_WRAPPER_H__
#define __OPENMP_CUDA_HOST_WRAPPER_H__

#include "cuda.h"

#include <cstdint>
#include <cstdio>
#include <omp.h>

extern "C" {
int __tgt_kernel(int64_t device_id, const void *host_ptr, void **args,
                 int32_t grid_dim_x, int32_t grid_dim_y, int32_t grid_dim_z,
                 int32_t block_dim_x, int32_t block_dim_y, int32_t block_dim_z,
                 size_t shared_mem, void *stream);

struct __omp_kernel_t {
  dim3 __grid_size;
  dim3 __block_size;
  size_t __shared_memory;

  void* __stream;
};

static __omp_kernel_t __current_kernel;
#pragma omp threadprivate(__current_kernel);

inline unsigned __cudaPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                            size_t __shared_memory,
                                            void* __stream_ptr) {
  __omp_kernel_t __kernel = __current_kernel;
  __kernel.__stream = __stream_ptr;
  __kernel.__grid_size = __grid_size;
  __kernel.__block_size = __block_size;
  __kernel.__shared_memory = __shared_memory;
  return 0;
}

inline unsigned __cudaPopCallConfiguration(dim3 *__grid_size,
                                           dim3 *__block_size,
                                           size_t *__shared_memory,
                                           void *__stream) {
  __omp_kernel_t &__kernel = __current_kernel;
  *__grid_size = __kernel.__grid_size;
  *__block_size = __kernel.__block_size;
  *__shared_memory = __kernel.__shared_memory;
  *((void**)__stream) = __kernel.__stream;
  return 0;
}

inline cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                    dim3 blockDim, void **args,
                                    size_t sharedMem, cudaStream_t stream) {
  __omp_kernel_t &__kernel = __current_kernel;

  int rv = __tgt_kernel(omp_get_default_device(), func, args, gridDim.x, gridDim.y,
               gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem,
               stream);
  return cudaError_t(rv);
}
}

#endif
