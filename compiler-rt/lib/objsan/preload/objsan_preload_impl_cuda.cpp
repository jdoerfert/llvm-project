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

#define CUDA_CHECK(Ret)                                                        \
  do {                                                                         \
    gpuAssert((Ret), __FILE__, __LINE__);                                      \
  } while (0)
namespace {
void gpuAssert(cudaError_t Ret, const char *File, int Line) {
  if (Ret != cudaSuccess) {
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(Ret), File,
            Line);
    exit(Ret);
  }
}
} // namespace

extern "C" {

__device__ char *__objsan_register_object(char *MPtr, uint64_t ObjSize,
                                          bool RequiresTemporalCheck);

__device__ void __objsan_free_object(char *VPtr);

__device__ void *__objsan_decode(char *VPtr);

__attribute__((used)) __global__
void __objsan_register_kernel(void **VPtr, void *MPtr, size_t Size) {
  *VPtr = __objsan_register_object(reinterpret_cast<char *>(MPtr), Size,
                                   /*RequiresTemporalCheck=*/false);
}

__attribute__((used)) __global__ void __objsan_unregister_kernel(void **MPtr,
                                                                 void *VPtr) {
  *MPtr = __objsan_decode(reinterpret_cast<char *>(VPtr));
  __objsan_free_object(reinterpret_cast<char *>(VPtr));
}

}; // extern "C"

namespace {

bool allocateDeviceMemory(void **DevPtr, size_t Size) {
  using FuncTy = cudaError_t(void **, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaMalloc");
  auto Err = FPtr(DevPtr, Size);
  if (Err != cudaSuccess) {
    fprintf(stderr, "%s CUDA error at %s:%d: %s\n",
            WarnPrefix, __FILE__, __LINE__, cudaGetErrorString(Err));
    return true;
  }
  return false;
}

bool freeDeviceMemory(void *DevPtr) {
  using FuncTy = cudaError_t(void *);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaFree");
  auto Err = FPtr(DevPtr);
  if (Err != cudaSuccess) {
    fprintf(stderr, "%s CUDA error at %s:%d: %s\n",
            WarnPrefix, __FILE__, __LINE__, cudaGetErrorString(Err));
    return true;
  }
  return false;
}

bool copyDeviceMemory(void *DstPtr, const void *SrcPtr, size_t Size,
                      cudaMemcpyKind Kind) {
  using FuncTy = cudaError_t(void *, const void *, size_t, cudaMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaMemcpy");
  auto Err = FPtr(DstPtr, SrcPtr, Size, Kind);
  if (Err != cudaSuccess) {
    fprintf(stderr, "%s CUDA error at %s:%d: %s\n",
            WarnPrefix, __FILE__, __LINE__, cudaGetErrorString(Err));
    return true;
  }
  return false;
}

bool deviceSynchronize() {
  using FuncTy = cudaError_t(void);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaDeviceSynchronize");
  auto Err = FPtr();
  if (Err != cudaSuccess) {
    fprintf(stderr, "%s CUDA error at %s:%d: %s\n",
            WarnPrefix, __FILE__, __LINE__, cudaGetErrorString(Err));
    return true;
  }
  return false;
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
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  void *VPtr = nullptr;
  auto Err =
      copyDeviceMemory(&VPtr, DevPtr, sizeof(void *), cudaMemcpyDeviceToHost);
  if (Err)
    fprintf(stderr, "error copying data from a registration\n");
  freeDeviceMemory(DevPtr);

  DPRINTF("%s registered mptr %p vptr %p size %zu\n", InfoPrefix, MPtr, VPtr,
          Size);

  return (Err) ? nullptr : VPtr;
}

void *launchUnregisterKernel(void *VPtr) {
  if (!VPtr)
    return nullptr;

  void **DevPtr;
  if (allocateDeviceMemory(reinterpret_cast<void **>(&DevPtr), sizeof(void *)))
    return nullptr;

  __objsan_unregister_kernel<<<1, 1>>>(DevPtr, VPtr);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  void *MPtr = nullptr;
  auto Err =
      copyDeviceMemory(&MPtr, DevPtr, sizeof(void *), cudaMemcpyDeviceToHost);
  if (Err)
    fprintf(stderr, "error copying data from a unregistration\n");
  freeDeviceMemory(DevPtr);

  DPRINTF("%s unregistered mptr %p vptr %p\n", InfoPrefix, MPtr, VPtr);

  return (Err) ? nullptr : MPtr;
}
} // namespace impl
} // namespace objsan

extern "C" {
using CtorFn = void (*)(void);
__attribute__((weak)) extern CtorFn __start___objsan_cuda_ctor;
__attribute__((weak)) extern CtorFn __stop___objsan_cuda_ctor;
}

namespace {
__attribute__((constructor(1000))) void __objsan_cuda_ctor_init() {
  if (&__start___objsan_cuda_ctor != nullptr) {
    // TODO Do we need to run the ctors on all devices?
    DPRINTF("Found cuda ctors at %p to %p\n", &__start___objsan_cuda_ctor,
            &__stop___objsan_cuda_ctor);
    for (CtorFn *Ctor = &__start___objsan_cuda_ctor,
                *E = &__stop___objsan_cuda_ctor;
         Ctor != E; ++Ctor) {
      DPRINTF("Calling device ctor at %p\n", Ctor);
      CUDA_CHECK(cudaLaunchKernel((const void *)*Ctor, dim3(1), dim3(1), nullptr, 0, nullptr));
#ifdef OBJSAN_DEBUG
      CUDA_CHECK(cudaDeviceSynchronize());
#endif
    }
#ifndef OBJSAN_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
  }
}
} // namespace
