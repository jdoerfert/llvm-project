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

#include "objsan_interface_internal.h"
#include "objsan_preload.h"

static __objsan::StatusTy *Status = nullptr;

__objsan::StatusTy *objsan::getStatus() {
  return Status;
}

void __objsan_rt_init(void) {
  objsan::impl::initialize(&Status);
}

void __objsan_rt_deinit(void) {
  objsan::impl::finalize(Status);
}
