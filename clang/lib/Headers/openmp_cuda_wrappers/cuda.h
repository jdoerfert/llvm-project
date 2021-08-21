/*===---- cuda.h - CUDA runtime support for OpenMP -----------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_OPENMP_FROM_CUDA_H__
#define __CLANG_OPENMP_FROM_CUDA_H__

#include "__openmp_cuda_common_wrapper.h"
#include "__openmp_cuda_host_wrapper.h"

#include <omp.h>

extern "C" {
void __omp_wait_for_stream(struct CUstream_st *__stream_ptr = 0);
}

using cudaError_t = int;
static constexpr cudaError_t cudaSuccess = 0;

inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char *cudaGetErrorString(cudaError_t) { return nullptr; }
inline void cudaGetDeviceCount(int *num_devices) {
  *num_devices = omp_get_num_devices();
}
inline void cudaSetDevice(int device) {
  omp_set_default_device(device);
}

/// used in cudaMemcpy to specify the copy direction
enum cudaMemcpyDir {
  cudaMemcpyHostToDevice, // From Host to Device
  cudaMemcpyDeviceToHost  // From Device to Host
};

/// Allocate memory on device. Takes a device pointer reference and size
template <typename Ty> inline void cudaMalloc(Ty **devicePtr, size_t size) {
  __omp_wait_for_stream();
  *devicePtr = (Ty *)omp_target_alloc(size, omp_get_default_device());
}

/// Copy memory from host to device or device to host.
template <typename Ty>
inline void cudaMemcpy(Ty *dst, Ty *src, size_t length,
                       cudaMemcpyDir direction) {
  // get the host device number (which is the initial device)
  int host_device_num = omp_get_initial_device();

  // use default device for gpu
  int gpu_device_num = omp_get_default_device();

  // default to copy from host to device
  int dst_device_num = gpu_device_num;
  int src_device_num = host_device_num;
  if (direction == cudaMemcpyDeviceToHost) {
    // copy from device to host
    dst_device_num = host_device_num;
    src_device_num = gpu_device_num;
  }

  __omp_wait_for_stream();
  // parameters are now set, call omp_target_memcpy
  omp_target_memcpy(dst, src, length, 0, 0, dst_device_num, src_device_num);
}

inline void cudaThreadSynchronize() { __omp_wait_for_stream(); }
inline void cudaDeviceSynchronize() { cudaThreadSynchronize(); }

/// Free allocated memory on device. Takes a device pointer
template <typename Ty> inline void cudaFree(Ty *devicePtr) {
  __omp_wait_for_stream();
  omp_target_free(devicePtr, omp_get_default_device());
}

#endif
