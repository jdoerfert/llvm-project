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
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>

namespace {

// A TLB that translates from MPtr to VPtr.
class TranslationLookasideBuffer {
  std::unordered_map<void *, void *> Map;
  std::mutex Lock;

public:
  bool insert(void *VPtr, void *MPtr) {
    assert(VPtr && "vptr is nullptr");
    assert(MPtr && "mptr is nullptr");
    std::lock_guard<std::mutex> LG(Lock);
    return Map.try_emplace(MPtr, VPtr).second;
  }

  void *translate(const void *MPtr) {
    if (!MPtr)
      return nullptr;
    std::lock_guard<std::mutex> LG(Lock);
    auto Itr = Map.find(const_cast<void *>(MPtr));
    return Itr == Map.end() ? nullptr : Itr->second;
  }

  void *pop(void *MPtr) {
    if (!MPtr)
      return nullptr;
    std::lock_guard<std::mutex> LG(Lock);
    auto Itr = Map.find(MPtr);
    if (Itr == Map.end())
      return nullptr;
    void *P = Itr->second;
    Map.erase(Itr);
    return P;
  }
} TLB;

} // namespace

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
    [[maybe_unused]] bool R = TLB.insert(devPtr, MPtr);
    assert(R && "a vptr has already existed");
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
    [[maybe_unused]] bool R = TLB.insert(devPtr, MPtr);
    assert(R && "a vptr has already existed");
    *devPtr = MPtr;
  }
  return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  void *MPtrFromTLB = TLB.pop(devPtr);
  void *MPtrFromDevice = objsan::unregisterDeviceMemory(devPtr);
  if (MPtrFromTLB == MPtrFromDevice) {
    devPtr = MPtrFromTLB;
  } else {
    fprintf(stderr, "mptr mismatch\n");
    if (MPtrFromDevice)
      devPtr = MPtrFromDevice;
    else if (MPtrFromTLB)
      devPtr = MPtrFromTLB;
  }
  using FuncTy = cudaError_t(void *);
  static FuncTy *FPtr = nullptr;
  if (!FPtr)
    FPtr = reinterpret_cast<FuncTy *>(objsan::getOriginalFunction("cudaFree"));
  assert(FPtr && "null cudaFree pointer");
  return FPtr(devPtr);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
  void *Dst = TLB.translate(dst);
  const void *Src = TLB.translate(src);
  if (!Dst)
    Dst = dst;
  if (!Src)
    Src = src;

  using FuncTy = cudaError_t(void *, const void *, size_t, cudaMemcpyKind);
  static FuncTy *FPtr = nullptr;
  if (!FPtr)
    FPtr =
        reinterpret_cast<FuncTy *>(objsan::getOriginalFunction("cudaMemcpy"));
  assert(FPtr && "null cudaMemcpy pointer");
  return FPtr(Dst, Src, count, kind);
}
