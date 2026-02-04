//===--------------- objsan_preload_impl_hip.cu ----------------*- C++ -*-===//
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

#include <hip/hip_runtime.h>

extern "C" {

__device__ char *
__objsan_register_object(char *MPtr, uint64_t ObjSize,
                         bool RequiresTemporalCheck);

__device__ void __objsan_free_object(char *VPtr);

__device__ void *__objsan_decode(char *VPtr);

__attribute__((used)) __global__
void __objsan_register_kernel(void **VPtr, void *MPtr, size_t Size) {
  *VPtr = __objsan_register_object(reinterpret_cast<char *>(MPtr), Size,
                                   /*RequiresTemporalCheck=*/false);
}

__attribute__((used)) __global__
void __objsan_unregister_kernel(void **MPtr, void *VPtr) {
  *MPtr = __objsan_decode(reinterpret_cast<char *>(VPtr));
  __objsan_free_object(reinterpret_cast<char *>(VPtr));
}

}; // extern "C"

namespace {

bool allocateDeviceMemory(void **DevPtr, size_t Size) {
  using FuncTy = hipError_t(void **, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("hipMalloc");
  return (FPtr(DevPtr, Size) != hipSuccess);
}

bool freeDeviceMemory(void *DevPtr) {
  using FuncTy = hipError_t(void *);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("hipFree");
  return (FPtr(DevPtr) != hipSuccess);
}

bool copyDeviceMemory(void *DstPtr, const void *SrcPtr, size_t Size,
                      hipMemcpyKind Kind) {
  using FuncTy = hipError_t(void *, const void *, size_t, hipMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("hipMemcpy");
  return (FPtr(DstPtr, SrcPtr, Size, Kind) != hipSuccess);
}

} // namespace

namespace objsan {
namespace impl {

void *launchRegisterKernel(void *MPtr, size_t Size) {
  if (!MPtr)
    return nullptr;

  void **DevPtr;
  if (allocateDeviceMemory(reinterpret_cast<void **>(&DevPtr), sizeof(void *)))
    return nullptr;

  __objsan_register_kernel<<<1, 1>>>(DevPtr, MPtr, Size);

  void *VPtr = nullptr;
  auto Err =
      copyDeviceMemory(&VPtr, DevPtr, sizeof(void *), hipMemcpyDeviceToHost);
  if (Err)
    fprintf(stderr, "error copying data from a registration\n");
  freeDeviceMemory(DevPtr);

  printf("%s registered mptr %p vptr %p size %zu\n", InfoPrefix, MPtr, VPtr, Size);

  return (Err) ? nullptr : VPtr;
}

void *launchUnregisterKernel(void *VPtr) {
  if (!VPtr)
    return nullptr;

  void **DevPtr;
  if (allocateDeviceMemory(reinterpret_cast<void **>(&DevPtr), sizeof(void *)))
    return nullptr;

  __objsan_unregister_kernel<<<1, 1>>>(DevPtr, VPtr);

  void *MPtr = nullptr;
  auto Err =
      copyDeviceMemory(&MPtr, DevPtr, sizeof(void *), hipMemcpyDeviceToHost);
  if (Err)
    fprintf(stderr, "error copying data from a unregistration\n");
  freeDeviceMemory(DevPtr);

  printf("%s unregistered mptr %p vptr %p\n", InfoPrefix, MPtr, VPtr);

  return (Err) ? nullptr : MPtr;
}
} // namespace impl
} // namespace objsan
