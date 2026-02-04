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

// A TLB that translates from VPtr to MPtr.
objsan::TLBTy TLB;

} // namespace

hipError_t hipMalloc(void **devPtr, size_t size) {
  using FuncTy = hipError_t(void **, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  assert(FPtr && "null hipMalloc pointer");

  hipError_t Err = FPtr(devPtr, size);
  if (Err != hipSuccess)
    return Err;
  void *VPtr = objsan::registerDeviceMemory(*devPtr, size);
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
  assert(FPtr && "null hipMallocManaged pointer");
  hipError_t Err = FPtr(devPtr, size, flags);
  if (Err != hipSuccess)
    return Err;
  void *VPtr = objsan::registerDeviceMemory(*devPtr, size);
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
  void *MPtrFromDev = objsan::unregisterDeviceMemory(devPtr);
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
  assert(FPtr && "null hipFree pointer");
  return FPtr(devPtr);
}

hipError_t hipMemcpy(void *dst, const void *src, size_t count,
                     hipMemcpyKind kind) {
  void *Dst = TLB.translate(dst);
  const void *Src = TLB.translate(src);

  using FuncTy = hipError_t(void *, const void *, size_t, hipMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  assert(FPtr && "null hipMemcpy pointer");
  return FPtr(Dst, Src, count, kind);
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t count,
                       hipMemcpyKind kind, hipStream_t stream) {
  void *Dst = TLB.translate(dst);
  const void *Src = TLB.translate(src);

  using FuncTy = hipError_t(void *, const void *, size_t, hipMemcpyKind, hipStream_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  assert(FPtr && "null hipMemcpyAsync pointer");
  return FPtr(Dst, Src, count, kind, stream);
}

hipError_t hipMemset(void *dst, int value, size_t count) {
  void *Dst = TLB.translate(dst);

  using FuncTy = hipError_t(void *, int, size_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  assert(FPtr && "null hipMemset pointer");
  return FPtr(Dst, value, count);
}

hipError_t hipMemsetAsync(void *dst, int value, size_t count, hipStream_t stream) {
  void *Dst = TLB.translate(dst);

  using FuncTy = hipError_t(void *, int, size_t, hipStream_t);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  assert(FPtr && "null hipMemsetAsync pointer");
  return FPtr(Dst, value, count, stream);
}
