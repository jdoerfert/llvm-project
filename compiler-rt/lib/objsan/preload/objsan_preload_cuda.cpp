//===----------------- objsan_preload_cuda.cpp ------------------*- C++ -*-===//
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
#include <cstdio>

#include <cuda_runtime.h>

namespace {

cudaError_t checkStatus(cudaError_t Error);

} // namespace

#define CUDA_CHECK(Call)                                                       \
    do {                                                                       \
        cudaError_t Error = (Call);                                            \
        if (Error != cudaSuccess) {                                            \
            Error = checkStatus(Error);                                        \
            fprintf(stderr, "CUDA Error at %s:%d: %s (%d)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(Error), Error);     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

extern "C" {

__device__ char *__objsan_register_object(char *MPtr, uint64_t ObjSize,
                                          bool RequiresTemporalCheck);
__device__ void __objsan_free_object(char *VPtr);
__device__ void *__objsan_decode(char *VPtr);
__device__ void __objsan_setup_status(__objsan::StatusTy *Status);

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

__attribute__((used)) __global__
void __objsan_setup_status_kernel(__objsan::StatusTy *Status) {
  __objsan_setup_status(Status);
}

}; // extern "C"

namespace {

// A TLB that translates from VPtr to MPtr.
objsan::TLBTy TLB;

cudaError_t checkStatus(cudaError_t Error) {
  if (Error == cudaSuccess)
    return Error;

  auto *Status = objsan::getStatus();
  if (Status && Status->hasFailed())
    fprintf(stderr, "%s bad\n", Status->isLoad() ? "l" : "s");
  return Error;
}

cudaError_t allocDeviceMem(void **DevPtr, size_t Size) {
  using FuncTy = cudaError_t(void **, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaMalloc");
  return FPtr(DevPtr, Size);
}

cudaError_t freeDeviceMem(void *DevPtr) {
  using FuncTy = cudaError_t(void *);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaFree");
  return FPtr(DevPtr);
}

cudaError_t copyDeviceMem(void *DstPtr, const void *SrcPtr, size_t Size,
                      cudaMemcpyKind Kind) {
  using FuncTy = cudaError_t(void *, const void *, size_t, cudaMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("cudaMemcpy");
  return FPtr(DstPtr, SrcPtr, Size, Kind);
}

void *launchRegisterKernel(void *MPtr, size_t Size) {
  if (!MPtr)
    return nullptr;

  void **DevPtr;
  CUDA_CHECK(allocDeviceMem(reinterpret_cast<void **>(&DevPtr), sizeof(void *)));

  __objsan_register_kernel<<<1, 1>>>(DevPtr, MPtr, Size);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  void *VPtr = nullptr;
  CUDA_CHECK(copyDeviceMem(&VPtr, DevPtr, sizeof(void *), cudaMemcpyDeviceToHost));
  CUDA_CHECK(freeDeviceMem(DevPtr));

  DPRINTF("%s registered mptr %p vptr %p size %zu\n", InfoPrefix, MPtr, VPtr,
          Size);

  return VPtr;
}

void *launchUnregisterKernel(void *VPtr) {
  if (!VPtr)
    return nullptr;

  void **DevPtr;
  CUDA_CHECK(allocDeviceMem(reinterpret_cast<void **>(&DevPtr), sizeof(void *)));

  __objsan_unregister_kernel<<<1, 1>>>(DevPtr, VPtr);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  void *MPtr = nullptr;
  CUDA_CHECK(copyDeviceMem(&MPtr, DevPtr, sizeof(void *), cudaMemcpyDeviceToHost));
  CUDA_CHECK(freeDeviceMem(DevPtr));

  DPRINTF("%s unregistered mptr %p vptr %p\n", InfoPrefix, MPtr, VPtr);

  return MPtr;
}

} // namespace

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  using FuncTy = cudaError_t(void **, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);

  cudaError_t Err = checkStatus(FPtr(devPtr, size));
  if (Err != cudaSuccess)
    return Err;
  void *VPtr = launchRegisterKernel(*devPtr, size);
  if (!VPtr) {
    // emit warning but we can't fail here.
    fprintf(stderr, "failed to register device memory\n");
  } else {
    [[maybe_unused]] bool R = TLB.insert(*devPtr, VPtr);
    assert(R && "a vptr has already existed");
    *devPtr = VPtr;
  }
  return cudaSuccess;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
  using FuncTy = cudaError_t(void **, size_t, unsigned int);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);

  cudaError_t Err = checkStatus(FPtr(devPtr, size, flags));
  if (Err != cudaSuccess)
    return Err;
  void *VPtr = launchRegisterKernel(*devPtr, size);
  if (!VPtr) {
    // emit warning but we can't fail here.
    fprintf(stderr, "failed to register device memory\n");
  } else {
    [[maybe_unused]] bool R = TLB.insert(*devPtr, VPtr);
    assert(R && "a vptr has already existed");
    *devPtr = VPtr;
  }
  return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  void *MPtrFromTLB = TLB.pop(devPtr);
  void *MPtrFromDev = launchUnregisterKernel(devPtr);
  if (MPtrFromTLB == MPtrFromDev) {
    devPtr = MPtrFromTLB;
  } else {
    fprintf(stderr, "%s mismatch for vptr %p, mptr_tlb %p mptr_dev %p\n",
            WarnPrefix, devPtr, MPtrFromTLB, MPtrFromDev);
    if (MPtrFromDev)
      devPtr = MPtrFromDev;
    else if (MPtrFromTLB)
      devPtr = MPtrFromTLB;
  }
  using FuncTy = cudaError_t(void *);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(devPtr));
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
  void *Dst = TLB.translate(dst);
  const void *Src = TLB.translate(src);

  using FuncTy = cudaError_t(void *, const void *, size_t, cudaMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(Dst, Src, count, kind));
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind, cudaStream_t stream) {
  void *Dst = TLB.translate(dst);
  const void *Src = TLB.translate(src);

  using FuncTy = cudaError_t(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(Dst, Src, count, kind, stream));
}

cudaError_t cudaMemset(void *dst, int value, size_t count) {
  void *Dst = TLB.translate(dst);

  using FuncTy = cudaError_t(void *, int, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(Dst, value, count));
}

cudaError_t cudaMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream) {
  void *Dst = TLB.translate(dst);

  using FuncTy = cudaError_t(void *, int, size_t, cudaStream_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(Dst, value, count, stream));
}

cudaError_t cudaLaunchKernel(const void* function, dim3 nblocks, dim3 nthreads,
                           void** args, size_t sharedmem,
                           cudaStream_t stream) {
  using FuncTy = cudaError_t(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(function, nblocks, nthreads, args, sharedmem, stream));
}

cudaError_t cudaDeviceSynchronize(void) {
  using FuncTy = cudaError_t(void);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr());
}

cudaError_t cudaPeekAtLastError(void) {
  using FuncTy = cudaError_t(void);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr());
}

namespace objsan {
namespace impl {

void initialize(__objsan::StatusTy **Status) {
  if (!Status)
    return;

  CUDA_CHECK(cudaHostAlloc((void**)Status, sizeof(__objsan::StatusTy), cudaHostAllocMapped));
  new (*Status) __objsan::StatusTy();

  __objsan::StatusTy *StatusDev = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&StatusDev, *Status, 0));

  __objsan_setup_status_kernel<<<1, 1>>>(StatusDev);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void finalize(__objsan::StatusTy *Status) {
  if (Status)
    CUDA_CHECK(cudaFreeHost(Status));
}

} // namespace impl
} // namespace objsan

#if 0
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
#ifdef __OBJSAN_DEBUG__
      CUDA_CHECK(cudaDeviceSynchronize());
#endif
    }
#ifndef __OBJSAN_DEBUG__
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
  }
}
} // namespace
#endif
