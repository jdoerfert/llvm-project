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

void *objsan::registerDeviceMemory(void *MPtr, size_t Size) {
  return impl::launchRegisterKernel(MPtr, Size);
}

void *objsan::unregisterDeviceMemory(void *VPtr) {
  return impl::launchUnregisterKernel(VPtr);
}
