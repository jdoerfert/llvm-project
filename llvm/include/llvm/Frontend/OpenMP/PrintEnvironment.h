//===---- PrintEnvironment.h - OpenMP GPU print environment ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_PRINT_ENVIRONMENT_H
#define OMPTARGET_PRINT_ENVIRONMENT_H

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

#include "Environment.h"

#ifdef OMPTARGET_DEVICE_RUNTIME
namespace _OMP {
#endif

struct PrintSlotTy {
  union {
    struct {
      uint32_t FormatStringSize;
      uint32_t NumArgumentSlots;
    } Metadata;
    void *FormatString;
    void *Argument;
  } Payload;
};
static_assert(sizeof(PrintSlotTy) == sizeof(void *),
              "Size of print slot should match a pointer!");

/// The print environment is actually an array of print slots, pre-allocated by
/// the compiler. The size of the array is defined in the constant configuration
/// environment of the kernel.
struct alignas(alignof(PrintSlotTy)) PrintEnvironmentTy {
  /// The number of slots used already.
  uint32_t NumSlotsUsed;

  uint32_t Unused;

  PrintSlotTy *slots() {
    static_assert(sizeof(PrintEnvironmentTy) % alignof(PrintSlotTy) == 0,
                  "Size should match alignment!");
    return reinterpret_cast<PrintSlotTy *>(reinterpret_cast<char *>(this) +
                                           sizeof(PrintEnvironmentTy));
  }
};

#ifdef OMPTARGET_DEVICE_RUNTIME
} // namespace _OMP
#endif

#endif
