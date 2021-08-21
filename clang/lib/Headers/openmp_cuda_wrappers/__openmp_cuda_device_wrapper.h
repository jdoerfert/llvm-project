/*===---- __openmp_cuda_device_wrapper.h - CUDA device support for OpenMP --===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __OPENMP_CUDA_DEVICE_WRAPPER_H__
#define __OPENMP_CUDA_DEVICE_WRAPPER_H__

#include "__openmp_cuda_common_wrapper.h"

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

// The file implements built-in CUDA variables using __declspec(property).
// https://msdn.microsoft.com/en-us/library/yhfk0thd.aspx
// All read accesses of built-in variable fields get converted into calls to a
// getter function which in turn calls the appropriate builtin to fetch the
// value.

#define __CUDA_DEVICE_BUILTIN(FIELD, INTRINSIC)                                \
  __declspec(property(get = __fetch_builtin_##FIELD)) unsigned int FIELD;      \
  static inline __attribute__((always_inline))                                 \
  __attribute__((device)) unsigned int __fetch_builtin_##FIELD(void) {         \
    return INTRINSIC;                                                          \
  }

#if __cplusplus >= 201103L
#define __DELETE = delete
#else
#define __DELETE
#endif

__device__ int printf(const char *, ...);
__device__ int __kmpc_get_hardware_thread_id_in_block();
__device__ int GetBlockIdInKernel();
__device__ int __kmpc_get_hardware_num_blocks();
__device__ int __kmpc_get_hardware_num_threads_in_block();
__device__ unsigned GetWarpId();
__device__ unsigned GetWarpSize();
__device__ unsigned GetLaneId();
extern "C" __device__ void __syncthreads();

struct ident_t;
struct map_var_info_t;
int __tgt_target_teams_mapper(ident_t *loc, int64_t device_id, void *host_ptr,
                              int32_t arg_num, void **args_base, void **args,
                              int64_t *arg_sizes, int64_t *arg_types,
                              map_var_info_t *arg_names, void **arg_mappers,
                              int32_t num_teams, int32_t thread_limit);

__device__ int __get_thread_idx_x() {
  return __kmpc_get_hardware_thread_id_in_block() % block.__x;
}
__device__ int __get_thread_idx_y() {
  return (__kmpc_get_hardware_thread_id_in_block() / block.__x) % block.__y;
}
__device__ int __get_thread_idx_z() {
  return (__kmpc_get_hardware_thread_id_in_block() / block.__x) / block.__y;
}
__device__ int __get_block_idx_x() { return GetBlockIdInKernel() % grid.__x; }
__device__ int __get_block_idx_y() {
  return (GetBlockIdInKernel() / grid.__x) % grid.__y;
}
__device__ int __get_block_idx_z() {
  return (GetBlockIdInKernel() / grid.__x) / grid.__y;
}
__device__ int __get_block_dim_x() { return block.__x; }
__device__ int __get_block_dim_y() { return block.__y; }
__device__ int __get_block_dim_z() { return block.__z; }
__device__ int __get_grid_dim_x() { return grid.__x; }
__device__ int __get_grid_dim_y() { return grid.__y; }
__device__ int __get_grid_dim_z() { return grid.__z; }

// Make sure nobody can create instances of the special variable types.  nvcc
// also disallows taking address of special variables, so we disable address-of
// operator as well.
#define __CUDA_DISALLOW_BUILTINVAR_ACCESS(TypeName)                            \
  __attribute__((device)) TypeName() __DELETE;                                 \
  __attribute__((device)) TypeName(const TypeName &) __DELETE;                 \
  __attribute__((device)) void operator=(const TypeName &) const __DELETE;     \
  __attribute__((device)) TypeName *operator&() const __DELETE

struct __cuda_builtin_threadIdx_t : public uint3 {
  __CUDA_DEVICE_BUILTIN(x, __get_thread_idx_x());
  __CUDA_DEVICE_BUILTIN(y, __get_thread_idx_y());
  __CUDA_DEVICE_BUILTIN(z, __get_thread_idx_z());
  // threadIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_threadIdx_t);
};

struct __cuda_builtin_blockIdx_t : public uint3 {
  __CUDA_DEVICE_BUILTIN(x, __get_block_idx_x());
  __CUDA_DEVICE_BUILTIN(y, __get_block_idx_y());
  __CUDA_DEVICE_BUILTIN(z, __get_block_idx_z());
  // blockIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockIdx_t);
};

struct __cuda_builtin_blockDim_t : public dim3 {
  __CUDA_DEVICE_BUILTIN(x, __get_block_dim_x());
  __CUDA_DEVICE_BUILTIN(y, __get_block_dim_y());
  __CUDA_DEVICE_BUILTIN(z, __get_block_dim_z());
  // blockDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockDim_t);
};

struct __cuda_builtin_gridDim_t : public dim3 {
  __CUDA_DEVICE_BUILTIN(x, __get_grid_dim_x());
  __CUDA_DEVICE_BUILTIN(y, __get_grid_dim_y());
  __CUDA_DEVICE_BUILTIN(z, __get_grid_dim_z());
  // gridDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_gridDim_t);
};

[[clang::loader_uninitialized]] __device__ __cuda_builtin_threadIdx_t threadIdx;
[[clang::loader_uninitialized]] __device__ __cuda_builtin_blockIdx_t blockIdx;
[[clang::loader_uninitialized]] __device__ __cuda_builtin_blockDim_t blockDim;
[[clang::loader_uninitialized]] __device__ __cuda_builtin_gridDim_t gridDim;

#undef __CUDA_DEVICE_BUILTIN
#undef __CUDA_BUILTIN_VAR
#undef __CUDA_DISALLOW_BUILTINVAR_ACCESS
#undef __DELETE

#if defined(__cplusplus)
}
#endif

#endif

