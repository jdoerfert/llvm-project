//===--------------- objsan_preload_impl_cuda.cpp ---------------*- C++ -*-===//
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

extern __device__ char *__objsan_register_object(char *MPtr, uint64_t ObjSize,
                                                 bool RequiresTemporalCheck);

namespace {

__global__ void registerKernel(void *MPtr, void *VPtr, size_t Size) {
  void *P = __objsan_register_object(reinterpret_cast<char *>(VPtr), Size,
                                     /*RequiresTemporalCheck=*/false);
  *reinterpret_cast<void **>(MPtr) = P;
}

__global__ void unregisterKernel(void *VPtr, void *MPtr) {
  // to be continue...
}

void *allocateDeviceMemory(size_t Size) {
  using FuncTy = cudaError_t (*)(void **, size_t);
  static FuncTy FPtr = nullptr;
  if (!FPtr)
    FPtr = reinterpret_cast<FuncTy>(objsan::getOriginalFunction("cudaMalloc"));
  if (!FPtr) {
    // FIXME: some error message
    return nullptr;
  }
  void *DevPtr = nullptr;
  cudaError_t Err = FPtr(&DevPtr, sizeof(void *));
  if (Err != cudaSuccess)
    return nullptr;
  return DevPtr;
}

void freeDeviceMemory(void *Ptr) {
  using FuncTy = cudaError_t (*)(void *);
  static FuncTy FPtr = nullptr;
  if (!FPtr)
    FPtr = reinterpret_cast<FuncTy>(objsan::getOriginalFunction("cudaFree"));
  if (!FPtr) {
    // FIXME: some error message
    return;
  }
  (void)FPtr(Ptr);
}

} // namespace

namespace objsan {
namespace impl {
void initializeSupportedFunctionList(std::vector<const char *> &List) {
  List = {"cudaMalloc", "cudaMallocManaged", "cudaFree", "cudaMemcpy"};
}

bool launchRegisterKernel(void **MPtr, void *VPtr, size_t Size) {
  if (!MPtr)
    return false;

  void *DevPtr = allocateDeviceMemory(sizeof(void *));
  if (!DevPtr)
    return false;

  registerKernel<<<1, 1>>>(DevPtr, VPtr, Size);

  cudaError_t Err =
      cudaMemcpy(MPtr, DevPtr, sizeof(void *), cudaMemcpyDeviceToHost);
  freeDeviceMemory(DevPtr);
  if (Err != cudaSuccess) {
    *MPtr = VPtr;
    return false;
  }

  return true;
}

bool launchUnregisterKernel(void **VPtr, void *MPtr) {
  if (!VPtr)
    return false;

  void *DevPtr = allocateDeviceMemory(sizeof(void *));
  if (!DevPtr)
    return false;

  unregisterKernel<<<1, 1>>>(DevPtr, VPtr);
  cudaError_t Err =
      cudaMemcpy(VPtr, DevPtr, sizeof(void *), cudaMemcpyDeviceToHost);
  freeDeviceMemory(DevPtr);
  if (Err != cudaSuccess) {
    *VPtr = MPtr;
    return false;
  }

  return true;
}
} // namespace impl
} // namespace objsan
