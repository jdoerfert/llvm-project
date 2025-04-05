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

// Device compilation special handling headers
#ifndef __OBJSAN_DEVICE__

#include <stddef.h>
#include <stdint.h>

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

typedef signed long int int64_t;
typedef unsigned long int uint64_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed short int int16_t;
typedef unsigned short int uint16_t;
typedef signed char int8_t;
typedef unsigned char uint8_t;

// FIXME unsure if this is correct???
typedef uint64_t size_t;
typedef uint64_t intptr_t;

#define PRIu64 "lu"
#define PRId64 "ld"

extern "C" {
int printf (const char *__restrict __format, ...);
int vprintf(const char *format, va_list vlist);

static inline int gpu_printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  int result = vprintf(format, args);
  va_end(args);
  return result;
}

static inline void __assert_fail(const char *expr, const char *file,
                                 unsigned line, const char *function) {
  gpu_printf("%s:%u: %s: Assertion `%s` failed.\n", file, line, function, expr);
  __builtin_trap();
}
}

#define FPRINTF(...) gpu_printf(__VA_ARGS__)
#define FFLUSH(...)

#ifdef NDEBUG
#define __ASSERT_VOID_CAST static_cast<void>
#define assert(expr) (__ASSERT_VOID_CAST(0))
#else
static inline void __assert_fail(const char *expr, const char *file,
                                 unsigned line, const char *function) {
  printf("%s:%u: %s: Assertion `%s` failed.\n", file, line, function, expr);
  __builtin_trap();
}
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
