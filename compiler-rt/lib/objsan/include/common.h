//===------------------------ common.h --------------------------*- C++ -*-===//
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

#ifndef OBJSAN_INCLUDE_COMMON_H
#define OBJSAN_INCLUDE_COMMON_H

// Freestanding headers
#include <stdarg.h>

// Device compilation special handling headers
#ifndef __OBJSAN_DEVICE__

#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#define FPRINTF(...) fprintf(stderr, __VA_ARGS__)
#define FFLUSH(s) fflush((s))

#else

#define PRIu64 "llu"
#define PRId64 "lld"

using int8_t = char;
using uint8_t = unsigned char;

using int16_t = short;
using uint16_t = unsigned short;

using int32_t = int;
using uint32_t = unsigned;

using int64_t = long long int;
using uint64_t = long long unsigned;

using intptr_t = int64_t;
using size_t = uint64_t;

static_assert(sizeof(int8_t) == 1, "int8_t size mismatch");
static_assert(sizeof(int16_t) == 2, "int16_t size mismatch");
static_assert(sizeof(int32_t) == 4, "int32_t size mismatch");
static_assert(sizeof(int64_t) == 8, "uint64_t size mismatch");

extern "C" {
int printf(const char *format, ...);

static inline void __assert_fail(const char *expr, const char *file,
                                 unsigned line, const char *function) {
  printf("%s:%u: %s: Assertion `%s` failed.\n", file, line, function, expr);
  __builtin_trap();
}
}

#define FPRINTF(...) printf(__VA_ARGS__)
#define FFLUSH(...)

#ifdef NDEBUG
#define assert(expr) ((void)(0))
#else
#define assert(expr)                                                           \
  {                                                                            \
    if (!(expr))                                                               \
      __assert_fail(#expr, __FILE__, __LINE__, __PRETTY_FUNCTION__);           \
  }
#endif

namespace std {

template <typename T1, typename T2> struct pair {
  T1 first;
  T2 second;

  template <typename U1, typename U2>
  pair(U1 First, U2 Second) : first(First), second(Second) {}
};

} // namespace std

#endif

namespace __objsan {

enum OrderingTy {
  relaxed = __ATOMIC_RELAXED,
  aquire = __ATOMIC_ACQUIRE,
  release = __ATOMIC_RELEASE,
  acq_rel = __ATOMIC_ACQ_REL,
  seq_cst = __ATOMIC_SEQ_CST,
};

enum MemScopeTy {
  system = __MEMORY_SCOPE_SYSTEM,
  device = __MEMORY_SCOPE_DEVICE,
  workgroup = __MEMORY_SCOPE_WRKGRP,
  wavefront = __MEMORY_SCOPE_WVFRNT,
  single = __MEMORY_SCOPE_SINGLE,
};

} // namespace __objsan

#endif // OBJSAN_INCLUDE_COMMON_H
