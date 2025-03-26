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

#include <cstddef>
#include <vector>

namespace objsan {

namespace impl {

void initializeSupportedFunctionList(std::vector<const char *> &List);

bool launchRegisterKernel(void **MPtr, void *VPtr, size_t Size);

bool launchUnregisterKernel(void **VPtr, void *MPtr);

} // namespace impl

void *getOriginalFunction(const char *Name);

void *registerDeviceMemory(void *VPtr, size_t Size);

void *unregisterDeviceMemory(void *Ptr);

} // namespace objsan

#endif
