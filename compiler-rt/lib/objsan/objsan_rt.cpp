//===- objsan/objsan_rt.cpp -----------------------------------------------===//
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

#include "include/obj_encoding.h"
#include "include/objsan_utils.h"

extern "C" {

__attribute__((
    visibility("default"))) __objsan::SmallObjectsTy __objsan_SmallObjects;
__attribute__((
    visibility("default"))) __objsan::LargeObjectsTy __objsan_LargeObjects;
};

__attribute__((visibility("default"))) __objsan::StatusTy *__objsan_Status = nullptr;

namespace __objsan {

#ifdef STATS
#ifndef __OBJSAN_DEVICE__
__attribute__((visibility("default"))) StatsTy SLoads("loads");
__attribute__((visibility("default"))) StatsTy SStores("stores");
__attribute__((visibility("default"))) StatsTy SRange("range");
__attribute__((visibility("default"))) StatsTy SLoopR("loopr");
#endif
#endif

} // namespace __objsan

#ifdef __OBJSAN_USE_START_STOP_SECTION_CTOR__
extern "C" {
using CtorFn = void (*)(void);
__attribute__((weak)) extern CtorFn __start___objsan_ctor;
__attribute__((weak)) extern CtorFn __stop___objsan_ctor;

__attribute__((constructor(1000))) void __objsan_ctor_init() {
  //  fprintf(stderr, "CTOR INIT  %p %p, %lu\n", &__start___objsan_ctor,
  //          &__stop___objsan_ctor,
  //          &__stop___objsan_ctor - &__start___objsan_ctor);
  assert(&__start___objsan_ctor == nullptr && &__stop___objsan_ctor == nullptr);
  if (&__start___objsan_ctor != nullptr) {
    for (CtorFn *Ctor = &__start___objsan_ctor, *E = &__stop___objsan_ctor;
         Ctor != E; ++Ctor)
      (*Ctor)();
  }
}
}
#endif
