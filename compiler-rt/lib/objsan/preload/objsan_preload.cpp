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

#include "objsan_preload.h"

#include <mutex>
#include <unordered_map>

#include <dlfcn.h>

void *objsan::registerDeviceMemory(void *VPtr, size_t Size) {
  void *MPtr = nullptr;
  return impl::launchRegisterKernel(&MPtr, VPtr, Size) ? MPtr : VPtr;
}

void *objsan::unregisterDeviceMemory(void *MPtr) {
  void *VPtr = nullptr;
  return impl::launchUnregisterKernel(&MPtr, VPtr) ? VPtr : MPtr;
}

namespace {

std::once_flag FunctionMapOnceFlag;

std::unordered_map<const char *, void *> FunctionMap;

void init() {
  std::vector<const char *> List;
  objsan::impl::initializeSupportedFunctionList(List);
  for (const char *Name : List) {
    void *P = dlsym(RTLD_NEXT, Name);
    if (!P)
      FunctionMap[Name] = P;
  }
}

} // namespace

void *objsan::getOriginalFunction(const char *Name) {
  std::call_once(FunctionMapOnceFlag, init);
  auto Itr = FunctionMap.find(Name);
  if (Itr != FunctionMap.end())
    return Itr->second;
  return nullptr;
}
