//===-------------------- objsan_preload.cpp --------------------*- C++ -*-===//
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

#include <mutex>
#include <unordered_map>

void *objsan::registerDeviceMemory(void *VPtr, size_t Size) {
  void *MPtr = nullptr;
  return impl::launchRegisterKernel(&MPtr, VPtr, Size) ? MPtr : VPtr;
}

void *objsan::unregisterDeviceMemory(void *MPtr) {
  void *VPtr = nullptr;
  return impl::launchUnregisterKernel(&MPtr, VPtr) ? VPtr : MPtr;
}

namespace {

class FunctionMapTableTy {
  std::unordered_map<const char *, void *> Table;
  std::mutex TableLock;

public:
  void *get(const char *SymName) {
    std::lock_guard<std::mutex> LG(TableLock);
    auto Itr = Table.find(SymName);
    if (Itr != Table.end())
      return Itr->second;
    void *P = dlsym(RTLD_NEXT, SymName);
    if (!P)
      return nullptr;
    Table[SymName] = P;
    return P;
  }
} FunctionMapTable;
} // namespace

void *objsan::getOriginalFunction(const char *Name) {
  return FunctionMapTable.get(Name);
}
