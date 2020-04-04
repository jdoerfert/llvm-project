//===---- target.h - OpenMP defines and helpers for target code --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines and helpers for target code.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_TARGET_H
#define OMPTARGET_TARGET_H

#include <cstdint>

#define __p(STR) _Pragma(STR)
#define __p2(STR) __p(#STR)

#define __DEVICE_SCOPE_BEGIN()                                                 \
  extern "C" {                                                                 \
  __p("omp declare target")

#define __DEVICE_SCOPE_END()                                                   \
  __p("omp end declare target")                                                \
  } /* extern "C" */

#define __CONSTEXPR static constexpr __attribute__((nothrow, always_inline))

#define __LEAGUE_VAR(TYPE, NAME)                                               \
  TYPE NAME [[clang::loader_uninitialized]];                                   \
  __p2(omp declare target to(NAME))

#define __TEAM_VAR(TYPE, NAME)                                                 \
  TYPE NAME [[clang::loader_uninitialized]];                                   \
  __p2(omp allocate(NAME) allocator(omp_pteam_mem_alloc))                      \
      __p2(omp declare target to(NAME))

#define __THREAD_VAR(TYPE, NAME)                                               \
  TYPE NAME [[clang::loader_uninitialized]];                                   \
  __p2(omp allocate(NAME) allocator(omp_thread_mem_alloc))                     \
      __p2(omp declare target to(NAME))

#endif
