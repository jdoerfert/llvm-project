//===-------- Debug.h ---- Debug utilities ------------------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_DEBUG_H
#define OMPTARGET_DEVICERTL_DEBUG_H

#include "Configuration.h"
#include "DeviceEnvironment.h"
#include "Utils.h"

/// Assertion
///
/// {
extern "C" {
void __assert_assume(bool cond, bool enabled, bool print, const char *exp,
                     const char *file, int line);
void __assert_fail(const char *assertion, const char *file, unsigned line,
                   const char *function);
}

/// Helper expansion for ASSERT. \p Cond is the expression in the assertion,
/// \p Enabled determines if assertions are enabled at all, \p ID is a unique
/// identifier for the assertion so we can ensure it is only reported once.
#define ASSERT_IMPL(Cond, Enabled)                                             \
  __assert_assume(Cond, Enabled, Enabled &&utils::SingletonFlag::testAndSet(), \
                  #Cond, __FILE__, __LINE__)

/// Assert \p Cond holds. If assertions are enabled it will check it, otherwise
/// simply assume it holds.
#define ASSERT(Cond)                                                           \
  ASSERT_IMPL(Cond, config::isConfigurationEnabled(config::EnableAssertion))

///}

/// Print
/// TODO: For now we have to use macros to guard the code because Clang lowers
/// `printf` to different function calls on NVPTX and AMDGCN platforms, and it
/// doesn't work for AMDGCN. After it can work on AMDGCN, we will remove the
/// macro.
/// {

extern "C" {
int printf(const char *format, ...);
}

#define PRINTF(fmt, ...) (void)printf(fmt, __VA_ARGS__)
#define PRINT(str) PRINTF("%s", str)

#define WARN(fmt, ...) PRINTF("WARNING: " #fmt, __VA_ARGS__)

///}

/// Enter a debugging scope for performing function traces. Enabled with
/// FunctionTracting set in the debug kind.
#define FunctionTracingRAII()                                                  \
  DebugEntryRAII Entry(__LINE__, __PRETTY_FUNCTION__);

/// An RAII class for handling entries to debug locations. The current location
/// and function will be printed on entry. Nested levels increase the
/// indentation shown in the debugging output.
struct DebugEntryRAII {
  DebugEntryRAII(const unsigned Line, const char *Function);
  ~DebugEntryRAII();
};

#endif
