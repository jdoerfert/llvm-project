//===------ State.cpp - OpenMP State & ICV interface ------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Allocator.h"
#include "Configuration.h"
#include "Mapping.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;

#pragma omp begin declare target device_type(nohost)

uint64_t *CONSTANT(__omp_rtl_device_memory_tracker)
    __attribute__((used, retain, weak, visibility("protected")));

struct BumpAllocatorTy final {

  void init(void *Ptr, uint64_t Size) {
    Data = reinterpret_cast<uint64_t>(Ptr);
    End = Data + Size;
  }

  void *alloc(uint64_t Size) {
    atomic::add(&__omp_rtl_device_memory_tracker[0], 1, atomic::seq_cst);
    atomic::add(&__omp_rtl_device_memory_tracker[1], Size, atomic::seq_cst);
    atomic::min(&__omp_rtl_device_memory_tracker[2], Size, atomic::seq_cst);
    atomic::max(&__omp_rtl_device_memory_tracker[3], Size, atomic::seq_cst);
    Size = utils::roundUp(Size, uint64_t(allocator::ALIGNMENT));
    uint64_t OldData = atomic::add(&Data, Size, atomic::seq_cst);
    if (OldData + Size > End)
      __builtin_trap();
    return reinterpret_cast<void *>(OldData);
  }

  void free(void *) {}

private:
  uint64_t Data;
  uint64_t End;
};

static BumpAllocatorTy SHARED(BumpAllocator);

/// allocator namespace implementation
///
///{

void allocator::init(bool IsSPMD, KernelEnvironmentTy &KernelEnvironment) {
  // TODO: Check KernelEnvironment for an allocator choice as soon as we have
  // more than one.
  // TODO: Allow non initialized globals with non trivial constructors.
  if (mapping::isInitialThreadInLevel0(IsSPMD)) {
    BumpAllocator.init(config::getDeviceMemoryPoolPtr(),
                       config::getDeviceMemoryPoolSize());
  }
}

void *allocator::alloc(uint64_t Size) { return BumpAllocator.alloc(Size); }

void allocator::free(void *Ptr) { BumpAllocator.free(Ptr); }

///}

#pragma omp end declare target
