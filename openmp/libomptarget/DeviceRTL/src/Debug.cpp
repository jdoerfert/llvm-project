//===--- Debug.cpp -------- Debug utilities ----------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains debug utilities
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Configuration.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace _OMP;

#pragma omp declare target

extern "C" {
void __assert_assume(bool condition) { __builtin_assume(condition); }

void __assert_fail(const char *assertion, const char *file, unsigned line,
                   const char *function) {
  PRINTF("%s:%u: %s: Assertion `%s' failed.\n", file, line, function,
         assertion);
  __builtin_trap();
}

namespace impl {

static int32_t omp_vprintf(const char *Format, void *Arguments,
                           uint32_t ArgumentsSize) {
  uint32_t NumSlotsAvailable = config::getNumPrintSlots();
  ASSERT(NumSlotsAvailable > 0 && "OpenMP vprintf needs print slots!");

  PrintEnvironmentTy *PrintEnvironment =
      state::getKernelEnvironment().PrintEnvironment;
  ASSERT(PrintEnvironment &&
         "Print environment should not be null if print slots are available!");

  uint32_t NumArgumentSlots =
      (ArgumentsSize + sizeof(void *) - 1) / sizeof(void *);
  uint32_t NumSlotsNeeded =
      /* Metadata */ 1 + /* Format */ 1 + NumArgumentSlots;
  if (NumSlotsNeeded > NumSlotsAvailable)
    return -1;

  uint32_t FirstSlot =
      atomic::add(&PrintEnvironment->NumSlotsUsed, NumSlotsNeeded, atomic::SEQ_CST);
  if (FirstSlot + NumSlotsNeeded > NumSlotsAvailable) {
    // For now we fill the buffer and stop printing, wrap-around or
    // multi-buffer support comes later.
    atomic::add(&PrintEnvironment->NumSlotsUsed, -NumSlotsNeeded, atomic::SEQ_CST);
    return -1;
  }

  // We reserved [FirstSlot : FirstSlot + NumSlotsNeeded], write the
  // values now.
  PrintSlotTy *Slots = PrintEnvironment->slots();
  Slots[FirstSlot + 0].Payload.Metadata.FormatStringSize =
      utils::strlen(Format);
  Slots[FirstSlot + 0].Payload.Metadata.NumArgumentSlots = NumArgumentSlots;
  Slots[FirstSlot + 1].Payload.FormatString =
      const_cast<void *>(reinterpret_cast<const void *>(Format));
  utils::memcpy(&Slots[FirstSlot + 2], Arguments, ArgumentsSize);

  // TODO: We basically do not support the correct return value yet, it's hard
  // to do and probably not necessary (for now).
  return -1;
}

/// The "native" vprintf that is provided by the environment.
int32_t vprintf(const char *, void *);

/// Native vprintf just uses the vprintf provided by the environment.
static int32_t vprintf_native(const char *Format, void *Arguments, uint32_t) {
  return vprintf(Format, Arguments);
}

/// AMDGPU doesn't provide a native vprintf so we make it a no-op.
#pragma omp begin declare variant match(device = {arch(amdgcn)})
static int32_t vprintf_native(const char *, void *, uint32_t) { return -1; }
#pragma omp end declare variant

} // namespace impl

int32_t __llvm_omp_vprintf(const char *Format, void *Arguments,
                           uint32_t ArgumentsSize) {
  if (config::getNumPrintSlots())
    return impl::omp_vprintf(Format, Arguments, ArgumentsSize);
  return impl::vprintf_native(Format, Arguments, ArgumentsSize);
}
}

DebugEntryRAII::DebugEntryRAII(const char *File, const unsigned Line,
                               const char *Function) {
  if (config::isDebugMode(config::DebugKind::FunctionTracing) &&
      mapping::getThreadIdInBlock() == 0 && mapping::getBlockId() == 0) {
    uint16_t Level = state::getKernelEnvironment().DebugIndentionLevel;

    for (int I = 0; I < Level; ++I)
      PRINTF("%s", "  ");

    PRINTF("%s:%u: Thread %u Entering %s\n", File, Line,
           mapping::getThreadIdInBlock(), Function);
    Level++;
  }
}

DebugEntryRAII::~DebugEntryRAII() {
  if (config::isDebugMode(config::DebugKind::FunctionTracing) &&
      mapping::getThreadIdInBlock() == 0 && mapping::getBlockId() == 0) {
    uint16_t Level = state::getKernelEnvironment().DebugIndentionLevel;
    Level--;
  }
}

#pragma omp end declare target
