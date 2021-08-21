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

#include "__openmp_cuda_common_wrapper.h"
#include "__openmp_cuda_device_wrapper.h"
#include "cuda.h"

#if defined(__cplusplus)
#include <cstdint>
#include <cstdio>
#include <mutex>

extern "C" {
struct ident_t {
  int32_t a = 0;
  int32_t b = 2;
  int32_t c = 0;
  int32_t d = 0;
  const char *e = ";unknown;fopenmp-from-cuda;0;0;;";
} __ident;

struct map_var_info_t;
int __tgt_target_teams_nowait_mapper(
    ident_t *loc, int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t team_num,
    int32_t thread_limit, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList);

typedef struct kmp_task_t kmp_task_t;
typedef struct ident_t ident_t;
typedef int32_t kmp_int32;
typedef int64_t kmp_int64;
typedef int32_t(kmp_routine_entry_t)(int32_t, char *);

kmp_int32 __kmpc_omp_task(ident_t *loc_ref, kmp_int32 gtid,
                          kmp_task_t *new_task);
kmp_task_t *__kmpc_omp_target_task_alloc(
    ident_t *loc_ref, kmp_int32 gtid, kmp_int32 flags, size_t sizeof_kmp_task_t,
    size_t sizeof_shareds, kmp_routine_entry_t task_entry, kmp_int64 device_id);

struct __omp_stream_t;

struct __omp_kernel_t {
  static unsigned constexpr __max_args = 64;

  inline void __finalize();

  void *__arg_pointers[__max_args];
  long int __arg_sizes[__max_args];
  long int __arg_types[__max_args];
  unsigned __num_args = 0;

  dim3 __grid_size;
  dim3 __block_size;
  size_t __shared_memory;

  __omp_stream_t *__stream;
};

static __omp_kernel_t *__current_kernel = 0;
#pragma omp threadprivate(__current_kernel);

struct __omp_stream_t {
  static unsigned constexpr __max_streams = 32;
  static unsigned constexpr __max_kernels = 32;

  __omp_kernel_t __kernels[__max_kernels];
  unsigned __first_kernel = 0;
  unsigned __last_kernel = 0;

  std::mutex __mtx;
};

static __omp_stream_t __streams[__omp_stream_t::__max_streams];

void __omp_kernel_t::__finalize() {
  __num_args = 0;
  __stream->__mtx.unlock();
}

inline __omp_stream_t *__omp_get_stream(struct CUstream_st *__stream_ptr = 0) {
  uintptr_t __stream_num = uintptr_t(__stream_ptr);
  __stream_num = __stream_num % __omp_stream_t::__max_streams;
  __omp_stream_t &__stream = __streams[__stream_num];
  return &__stream;
}

inline void __omp_wait_for_stream(struct CUstream_st *__stream_ptr) {
  __omp_stream_t &__stream = *__omp_get_stream(__stream_ptr);
#pragma omp task if (0) depend(inout : __stream)
  {}
  (void)__stream;
}

inline unsigned
__libompxPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                               size_t __shared_memory = 0,
                               struct CUstream_st *__stream_ptr = 0) {
  __omp_stream_t &__stream = *__omp_get_stream(__stream_ptr);
  __stream.__mtx.lock();

  printf("push_config: grid: [%i,%i,%i], block: [%i, %i, %i], shared_mem: %lu, "
         "stream: %p\n",
         __grid_size.__x, __grid_size.__y, __grid_size.__z, __block_size.__x,
         __block_size.__y, __block_size.__z, __shared_memory,
         (void *)__stream_ptr);

  unsigned &__kernel_num = __stream.__last_kernel;
  unsigned __next_kernel_num =
      (__kernel_num + 1) % __omp_stream_t::__max_kernels;
  if (__stream.__first_kernel == __next_kernel_num) {
    printf("Too many kernels per stream, no wait support yet\n");
    __builtin_unreachable();
  }

  __omp_kernel_t __kernel = __stream.__kernels[__kernel_num];
  __kernel.__stream = &__stream;
  __current_kernel = &__kernel;
  __kernel.__grid_size = __grid_size;
  __kernel.__block_size = __block_size;
  __kernel.__shared_memory = __shared_memory;
  __kernel_num = __next_kernel_num;
  return 0;
}

inline int cudaSetupArgument(char *__vp, long int __size, long int __offset) {
  printf("setup arg: %p [%lu, %lu] %i : %p\n", (void *)__vp, __size, __offset,
         *(int *)__vp, *(void **)__vp);

  __omp_kernel_t &__kernel = *__current_kernel;
  char *__p = (char *)&__kernel.__arg_pointers[__kernel.__num_args];
  for (unsigned __i = 0; __i < __size; ++__i)
    __p[__i] = __vp[__i];
  __kernel.__arg_sizes[__kernel.__num_args] = __size;
  __kernel.__arg_types[__kernel.__num_args] = 288;
  ++__kernel.__num_args;
  return 0;
}

inline void __omp_hidden_helper_task(kmp_routine_entry_t Fn) {
  int tid = omp_get_thread_num();
  int dev = omp_get_initial_device();
  kmp_task_t *task = __kmpc_omp_target_task_alloc(
      &__ident, tid, 1, /* sizeof(kmp_task_t) */ 40, 0, Fn, dev);
  __kmpc_omp_task(&__ident, tid, task);
}

inline int cudaLaunch(void *fp) {
  __omp_kernel_t &__kernel = *__current_kernel;

  unsigned __num_teams = __kernel.__grid_size.__x * __kernel.__grid_size.__y *
                         __kernel.__grid_size.__z;
  unsigned __num_threads = __kernel.__block_size.__x *
                           __kernel.__block_size.__y *
                           __kernel.__block_size.__z;

#pragma omp target enter data map(always, to : grid, block)

  kmp_routine_entry_t *__lambda = [&](int32_t, char *) -> int32_t {
    return __tgt_target_teams_nowait_mapper(
        nullptr, -1, fp, __kernel.__num_args, __kernel.__arg_pointers,
        __kernel.__arg_pointers, __kernel.__arg_sizes, __kernel.__arg_types,
        nullptr, nullptr, __num_teams, __num_threads, 0, nullptr, 0, nullptr);
  };
  __omp_hidden_helper_task(__lambda);

  __kernel.__finalize();

  return 0;
}
}

#endif
#endif
