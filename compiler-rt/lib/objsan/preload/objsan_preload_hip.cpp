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
  void *MPtrFromDevice = objsan::unregisterDeviceMemory(devPtr);
  if (MPtrFromTLB == MPtrFromDevice) {
    devPtr = MPtrFromTLB;
  } else {
    if (MPtrFromDevice)
      devPtr = MPtrFromDevice;
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
  if (!Dst)
    Dst = dst;
  if (!Src)
    Src = src;

  using FuncTy = hipError_t(void *, const void *, size_t, hipMemcpyKind);
  static FuncTy *FPtr = objsan::getOriginalFunction<FuncTy>(__func__);
  assert(FPtr && "null hipMemcpy pointer");
  return FPtr(Dst, Src, count, kind);
}
