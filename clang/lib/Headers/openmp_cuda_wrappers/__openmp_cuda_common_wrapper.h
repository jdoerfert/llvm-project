/*===---- __openmp_cuda_common_wrapper.h - CUDA common support for OpenMP --===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __OPENMP_CUDA_COMMON_WRAPPER_H__
#define __OPENMP_CUDA_COMMON_WRAPPER_H__

#include <stddef.h>

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))

#define T3(type, name)                                                         \
  typedef struct name {                                                        \
    name(){};                                                                  \
    name(type _x, type _y = 1, type _z = 1) : __x(_x), __y(_y), __z(_z){};     \
    type __x;                                                                  \
    type __y;                                                                  \
    type __z;                                                                  \
  } name

T3(unsigned, dim3);
T3(unsigned, uint3);

#pragma omp begin declare target
__constant__ dim3 grid;
__constant__ dim3 block;
#pragma omp end declare target

#endif

