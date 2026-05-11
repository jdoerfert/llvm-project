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

#include "objsan_utils.h"

#ifdef __OBJSAN_DEBUG__
#define DPRINTF(...) fprintf(stderr, __VA_ARGS__)
#else
#define DPRINTF(...)                                                           \
  do {                                                                         \
  } while (0)
#endif

namespace objsan {
namespace impl {

void initialize(__objsan::StatusTy **Status);
void finalize(__objsan::StatusTy *Status);

} // namespace impl

// A TLB that translates from VPtr to MPtr.
class TLBTy {
  std::unordered_map<void *, void *> Map;
  std::mutex Lock;

public:
  /// Inserts the translation from \p VPtr to \p MPtr.
  bool insert(void *MPtr, void *VPtr) {
    assert(MPtr && "vptr is nullptr");
    assert(VPtr && "mptr is nullptr");
    std::lock_guard<std::mutex> LG(Lock);
    return Map.try_emplace(VPtr, MPtr).second;
  }

  /// Returns the translation of \p VPtr to MPtr, or returns \p VPtr if such
  /// pointer is not registered.
  void *translate(const void *VPtr) {
    if (!VPtr)
      return nullptr;
    std::lock_guard<std::mutex> LG(Lock);
    /// TODO: This should check partial overlapping.
    auto Itr = Map.find(const_cast<void *>(VPtr));
    return Itr == Map.end() ? const_cast<void *>(VPtr) : Itr->second;
  }

  /// Removes the translation for \p VPtr.
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

__objsan::StatusTy *getStatus();

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

constexpr const char *FailPrefix = "[preload-error]";
constexpr const char *WarnPrefix = "[preload-warn]";
constexpr const char *InfoPrefix = "[preload-info]";

#endif
