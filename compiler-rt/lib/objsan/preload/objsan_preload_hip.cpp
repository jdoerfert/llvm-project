//===------------------ objsan_preload_hip.cpp ------------------*- C++ -*-===//
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

#include <hip/hip_runtime.h>

namespace {

hipError_t checkStatus(hipError_t Error);

} // namespace

#define HIP_CHECK(Call)                                                        \
  do {                                                                         \
    hipError_t Error = (Call);                                                 \
    if (Error != hipSuccess) {                                                 \
      Error = checkStatus(Error);                                              \
      fprintf(stderr, "HIP Error at %s:%d: %s (%d)\n",                         \
              __FILE__, __LINE__, hipGetErrorString(Error), Error);            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

extern "C" {

__device__ char *
__objsan_register_object(char *MPtr, uint64_t ObjSize,
                         bool RequiresTemporalCheck);
__device__ void __objsan_free_object(char *VPtr);
__device__ void *__objsan_decode(char *VPtr);
__device__ void __objsan_setup_status(__objsan::StatusTy *Status);

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

__attribute__((used)) __global__
void __objsan_setup_status_kernel(__objsan::StatusTy *Status) {
  __objsan_setup_status(Status);
}

}; // extern "C"

namespace {

// A TLB that translates from VPtr to MPtr.
objsan::TLBTy TLB;

hipError_t checkStatus(hipError_t Error) {
  if (Error == hipSuccess)
    return Error;

  auto *Status = objsan::getStatus();
  if (Status && Status->hasFailed())
    fprintf(stderr, "%s bad\n", Status->isLoad() ? "l" : "s");
  return Error;
}

hipError_t allocDeviceMem(void **DevPtr, size_t Size) {
  using FuncTy = hipError_t(void **, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("hipMalloc");
  return FPtr(DevPtr, Size);
}

hipError_t freeDeviceMem(void *DevPtr) {
  using FuncTy = hipError_t(void *);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("hipFree");
  return FPtr(DevPtr);
}

hipError_t copyDeviceMem(void *DstPtr, const void *SrcPtr, size_t Size,
                      hipMemcpyKind Kind) {
  using FuncTy = hipError_t(void *, const void *, size_t, hipMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>("hipMemcpy");
  return FPtr(DstPtr, SrcPtr, Size, Kind);
}

void *launchRegisterKernel(void *MPtr, size_t Size) {
  if (!MPtr)
    return nullptr;

  void **DevPtr;
  HIP_CHECK(allocDeviceMem(reinterpret_cast<void **>(&DevPtr), sizeof(void *)));

  __objsan_register_kernel<<<1, 1>>>(DevPtr, MPtr, Size);

  void *VPtr = nullptr;
  HIP_CHECK(copyDeviceMem(&VPtr, DevPtr, sizeof(void *), hipMemcpyDeviceToHost));
  HIP_CHECK(freeDeviceMem(DevPtr));

  DPRINTF("%s registered mptr %p vptr %p size %zu\n", InfoPrefix, MPtr, VPtr, Size);

  return VPtr;
}

void *launchUnregisterKernel(void *VPtr) {
  if (!VPtr)
    return nullptr;

  void **DevPtr;
  HIP_CHECK(allocDeviceMem(reinterpret_cast<void **>(&DevPtr), sizeof(void *)));

  __objsan_unregister_kernel<<<1, 1>>>(DevPtr, VPtr);

  void *MPtr = nullptr;
  HIP_CHECK(copyDeviceMem(&MPtr, DevPtr, sizeof(void *), hipMemcpyDeviceToHost));
  HIP_CHECK(freeDeviceMem(DevPtr));

  DPRINTF("%s unregistered mptr %p vptr %p\n", InfoPrefix, MPtr, VPtr);

  return MPtr;
}

} // namespace

hipError_t hipMalloc(void **devPtr, size_t size) {
  using FuncTy = hipError_t(void **, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);

  hipError_t Err = checkStatus(FPtr(devPtr, size));
  if (Err != hipSuccess)
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
  return hipSuccess;
}

hipError_t hipMallocManaged(void **devPtr, size_t size, unsigned int flags) {
  using FuncTy = hipError_t(void **, size_t, unsigned int);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);

  hipError_t Err = checkStatus(FPtr(devPtr, size, flags));
  if (Err != hipSuccess)
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
  return hipSuccess;
}

hipError_t hipFree(void *devPtr) {
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
  using FuncTy = hipError_t(void *);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(devPtr));
}

hipError_t hipMemcpy(void *dst, const void *src, size_t count,
                     hipMemcpyKind kind) {
  void *Dst = TLB.translate(dst);
  const void *Src = TLB.translate(src);

  using FuncTy = hipError_t(void *, const void *, size_t, hipMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(Dst, Src, count, kind));
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t count,
                       hipMemcpyKind kind, hipStream_t stream) {
  void *Dst = TLB.translate(dst);
  const void *Src = TLB.translate(src);

  using FuncTy = hipError_t(void *, const void *, size_t, hipMemcpyKind, hipStream_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(Dst, Src, count, kind, stream));
}

hipError_t hipMemset(void *dst, int value, size_t count) {
  void *Dst = TLB.translate(dst);

  using FuncTy = hipError_t(void *, int, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(Dst, value, count));
}

hipError_t hipMemsetAsync(void *dst, int value, size_t count, hipStream_t stream) {
  void *Dst = TLB.translate(dst);

  using FuncTy = hipError_t(void *, int, size_t, hipStream_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(Dst, value, count, stream));
}

hipError_t hipLaunchKernel(const void* function, dim3 nblocks, dim3 nthreads,
                           void** args, size_t sharedmem,
                           hipStream_t stream) {
  using FuncTy = hipError_t(const void *, dim3, dim3, void **, size_t, hipStream_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr(function, nblocks, nthreads, args, sharedmem, stream));
}

hipError_t hipDeviceSynchronize(void) {
  using FuncTy = hipError_t(void);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr());
}

hipError_t hipPeekAtLastError(void) {
  using FuncTy = hipError_t(void);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  return checkStatus(FPtr());
}

namespace objsan {
namespace impl {

void initialize(__objsan::StatusTy **Status) {
  if (!Status)
    return;

  HIP_CHECK(hipHostMalloc((void**)Status, sizeof(__objsan::StatusTy), hipHostMallocMapped));
  new (*Status) __objsan::StatusTy();

  __objsan::StatusTy *StatusDev = nullptr;
  HIP_CHECK(hipHostGetDevicePointer((void**)&StatusDev, *Status, 0));

  __objsan_setup_status_kernel<<<1, 1>>>(StatusDev);
  HIP_CHECK(hipDeviceSynchronize());
}

void finalize(__objsan::StatusTy *Status) {
  if (Status)
    HIP_CHECK(hipHostFree(Status));
}

} // namespace impl
} // namespace objsan
