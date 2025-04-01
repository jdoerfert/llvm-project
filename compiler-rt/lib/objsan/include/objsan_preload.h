//===---------------------- objsan_preload.h --------------------*- C++ -*-===//
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

#ifndef OBJSAN_OBJSAN_PRELOAD_H
#define OBJSAN_OBJSAN_PRELOAD_H

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <unordered_map>

namespace objsan {

namespace impl {

void *launchRegisterKernel(void *MPtr, size_t Size);

void *launchUnregisterKernel(void *VPtr);

} // namespace impl

// A TLB that translates from VPtr to MPtr.
class TLBTy {
  std::unordered_map<void *, void *> Map;
  std::mutex Lock;

public:
  bool insert(void *MPtr, void *VPtr) {
    assert(MPtr && "vptr is nullptr");
    assert(VPtr && "mptr is nullptr");
    std::lock_guard<std::mutex> LG(Lock);
    return Map.try_emplace(VPtr, MPtr).second;
  }

  void *translate(const void *VPtr) {
    if (!VPtr)
      return nullptr;
    std::lock_guard<std::mutex> LG(Lock);
    auto Itr = Map.find(const_cast<void *>(VPtr));
    return Itr == Map.end() ? nullptr : Itr->second;
  }

  void *pop(void *VPtr) {
    if (!VPtr)
      return nullptr;
    std::lock_guard<std::mutex> LG(Lock);
    auto Itr = Map.find(VPtr);
    if (Itr == Map.end())
      return nullptr;
    void *P = Itr->second;
    Map.erase(Itr);
    return P;
  }
};

void *registerDeviceMemory(void *MPtr, size_t Size);

void *unregisterDeviceMemory(void *Ptr);

template <typename FuncTy>
FuncTy *getOriginalFunction(const char *Name) {
  void *Symbol = dlsym(RTLD_NEXT, Name);
  if (!Symbol) {
    fprintf(stderr, "symbol %s not found\n", Name);
    abort();
  }
  return reinterpret_cast<FuncTy *>(Symbol);
}

} // namespace objsan

#endif
