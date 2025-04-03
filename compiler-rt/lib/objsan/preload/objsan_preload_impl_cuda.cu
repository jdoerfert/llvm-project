//===--------------- objsan_preload_impl_cuda.cu ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ObjSan.
//
//===----------------------------------------------------------------------===//

#include "objsan_preload.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

extern "C" __device__ char *
__objsan_register_object(char *MPtr, uint64_t ObjSize,
                         bool RequiresTemporalCheck);

extern "C" __device__ void
__objsan_free_object(char *VPtr);

namespace {

__global__ void registerKernel(void **VPtr, void *MPtr, size_t Size) {
  *VPtr = __objsan_register_object(reinterpret_cast<char *>(MPtr), Size,
                                     /*RequiresTemporalCheck=*/false);
}

__global__ void unregisterKernel(void **MPtr, void *VPtr) {
  __objsan_free_object(reinterpret_cast<char *>(VPtr));
  *MPtr = nullptr;
}

bool allocateDeviceMemory(void **DevPtr, size_t Size) {
  using FuncTy = cudaError_t(void **, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaMalloc");
  return (FPtr(DevPtr, Size) != cudaSuccess);
}

bool freeDeviceMemory(void *DevPtr) {
  using FuncTy = cudaError_t(void *);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaFree");
  return (FPtr(DevPtr) != cudaSuccess);
}

bool copyDeviceMemory(void *DstPtr, const void *SrcPtr, size_t Size, cudaMemcpyKind Kind) {
  using FuncTy = cudaError_t(void *, const void *, size_t, cudaMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaMemcpy");
  return (FPtr(DstPtr, SrcPtr, Size, Kind) != cudaSuccess);
}

} // namespace

namespace objsan {
namespace impl {

void *launchRegisterKernel(void *MPtr, size_t Size) {
  if (!MPtr)
    return nullptr;

  void **DevPtr;
  if (auto Err = allocateDeviceMemory(reinterpret_cast<void **>(&DevPtr), sizeof(void *)))
    return nullptr;

  registerKernel<<<1, 1>>>(DevPtr, MPtr, Size);

  void *VPtr = nullptr;
  auto Err = copyDeviceMemory(&VPtr, DevPtr, sizeof(void *), cudaMemcpyDeviceToHost);
  freeDeviceMemory(DevPtr);

  return (Err) ? nullptr : VPtr;
}

void *launchUnregisterKernel(void *VPtr) {
  if (!VPtr)
    return nullptr;

  void **DevPtr;
  if (auto Err = allocateDeviceMemory(reinterpret_cast<void **>(&DevPtr), sizeof(void *)))
    return nullptr;

  unregisterKernel<<<1, 1>>>(DevPtr, VPtr);

  void *MPtr = nullptr;
  auto Err = cudaMemcpy(&MPtr, DevPtr, sizeof(void *), cudaMemcpyDeviceToHost);
  freeDeviceMemory(DevPtr);

  return (Err) ? nullptr : MPtr;
}
} // namespace impl
} // namespace objsan
