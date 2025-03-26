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

#include "include/objsan_preload.h"

#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  using FuncTy = cudaError_t(void **, size_t);
  static FuncTy *FPtr = nullptr;
  if (!FPtr)
    FPtr =
        reinterpret_cast<FuncTy *>(objsan::getOriginalFunction("cudaMalloc"));
  assert(FPtr && "null cudaMalloc pointer");
  cudaError_t Err = FPtr(devPtr, size);
  if (Err != cudaSuccess)
    return Err;
  void *MPtr = objsan::registerDeviceMemory(*devPtr, size);
  if (!MPtr) {
    // emit warning but we can't fail here.
    fprintf(stderr, "failed to register device memory\n");
  } else {
    *devPtr = MPtr;
  }
  return cudaSuccess;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
  using FuncTy = cudaError_t(void **, size_t, unsigned int);
  static FuncTy *FPtr = nullptr;
  if (!FPtr)
    FPtr = reinterpret_cast<FuncTy *>(
        objsan::getOriginalFunction("cudaMallocManaged"));
  assert(FPtr && "null cudaMallocManaged pointer");
  cudaError_t Err = FPtr(devPtr, size, flags);
  if (Err != cudaSuccess)
    return Err;
  void *MPtr = objsan::registerDeviceMemory(*devPtr, size);
  if (!MPtr) {
    // emit warning but we can't fail here.
    fprintf(stderr, "failed to register device memory\n");
  } else {
    *devPtr = MPtr;
  }
  return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  void *MPtr = objsan::unregisterDeviceMemory(devPtr);
  if (!MPtr) {
    // emit warning but we can't fail here.
    fprintf(stderr, "failed to unregister device memory\n");
  } else {
    devPtr = MPtr;
  }
  using FuncTy = cudaError_t(void *);
  static FuncTy *FPtr = nullptr;
  if (!FPtr)
    FPtr = reinterpret_cast<FuncTy *>(objsan::getOriginalFunction("cudaFree"));
  assert(FPtr && "null cudaFree pointer");
  return FPtr(devPtr);
}
