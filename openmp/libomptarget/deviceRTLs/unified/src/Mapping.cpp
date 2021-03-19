//===------- Mapping.cpp - OpenMP device runtime mapping helpers -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Mapping.h"

#include "State.h"
#include "Utils.h"

using namespace _OMP;

//namespace {

#pragma omp declare target

/// Fallback implementations are missing to trigger a link time error.
/// Implementations for new devices, including the host, should go into a
/// dedicated begin/end declare variant.
///
///{

LaneMaskTy activemaskImpl();
LaneMaskTy lanemaskLTImpl();
LaneMaskTy lanemaskGTImpl();
uint32_t getThreadIdInWarpImpl();
uint32_t getThreadIdInBlockImpl();
uint32_t getBlockSizeImpl();
uint32_t getKernelSizeImpl();
uint32_t getBlockIdImpl();
uint32_t getNumberOfBlocksImpl();
uint32_t getNumberOfProcessorElementsImpl();
uint32_t getWarpIdImpl();
uint32_t getWarpSizeImpl();
uint32_t getNumberOfWarpsInBlockImpl();

///}

/// AMDGCN implementations
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

uint32_t getGridDim(uint32_t n, uint16_t d) {
  uint32_t q = n / d;
  return q + (n > q * d);
}
uint32_t getWorkgroupDim(uint32_t group_id, uint32_t grid_size,
                         uint16_t group_size) {
  uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}

LaneMaskTy activemaskImpl() { return __builtin_amdgcn_read_exec(); }
LaneMaskTy lanemaskLTImpl() {
  uint32_t Lane = mapping::getThreadIdInWarp();
  int64_t Ballot = mapping::activemask();
  uint64_t Mask = ((uint64_t)1 << Lane) - (uint64_t)1;
  return Mask & Ballot;
}
LaneMaskTy lanemaskGTImpl() {
  uint32_t Lane = mapping::getThreadIdInWarp();
  if (lane == (mapping::getWarpSize() - 1))
    return 0;
  int64_t Ballot = mapping::activemask();
  uint64_t Mask = (~((uint64_t)0)) << (Lane + 1);
  return Mask & Ballot;
}
uint32_t getThreadIdInWarpImpl() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
uint32_t getThreadIdInBlockImpl() { return __builtin_amdgcn_workitem_id_x(); }
uint32_t getBlockSizeImpl() {
  return getWorkgroupDim(__builtin_amdgcn_workgroup_id_x(),
                         __builtin_amdgcn_grid_size_x(),
                         __builtin_amdgcn_workgroup_size_x());
}
uint32_t getKernelSizeImpl() {
  return __builtin_amdgcn_grid_size_x();
}
uint32_t getBlockIdImpl() { return __builtin_amdgcn_workgroup_id_x(); }
uint32_t getNumberOfBlocksImpl() {
  return getGridDim(__builtin_amdgcn_grid_size_x(),
                    __builtin_amdgcn_workgroup_size_x());
}
uint32_t getNumberOfProcessorElementsImpl() {
  // TODO
  return mapping::getBlockSize();
}
uint32_t getWarpIdImpl() {
  return mapping::getThreadIdInBlock() / mapping::getWarpSize();
}
uint32_t getWarpSizeImpl() { return 64; }
uint32_t getNumberOfWarpsInBlockImpl() {
  return mapping::getBlockSize() / mapping::getWarpSize();
}

#pragma omp end declare variant
///}

/// NVPTX implementations
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

LaneMaskTy activemaskImpl() {
  unsigned int Mask;
  asm volatile("activemask.b32 %0;" : "=r"(Mask));
  return Mask;
}
LaneMaskTy lanemaskLTImpl() {
  __kmpc_impl_lanemask_t Res;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(Res));
  return Res;
}
LaneMaskTy lanemaskGTImpl() {
  __kmpc_impl_lanemask_t Res;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(Res));
  return Res;
}
uint32_t getThreadIdInWarpImpl() {
return mapping::getThreadIdInBlock() & (mapping::getWarpSize() - 1);
}
uint32_t getThreadIdInBlockImpl() {
return __nvvm_read_ptx_sreg_tid_x();
}
uint32_t getBlockSizeImpl() {
return __nvvm_read_ptx_sreg_ntid_x();
}
uint32_t getKernelSizeImpl() {
return __nvvm_read_ptx_sreg_nctaid_x();
}
uint32_t getBlockIdImpl() {
return __nvvm_read_ptx_sreg_ctaid_x();
}
uint32_t getNumberOfBlocksImpl() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}
uint32_t getNumberOfProcessorElementsImpl() {
  // TODO
  return mapping::getBlockSize();
}
uint32_t getWarpIdImpl() {
  return mapping::getThreadIdInBlock() / mapping::getWarpSize();
}
uint32_t getWarpSizeImpl() { return 32; }
uint32_t getNumberOfWarpsInBlockImpl() {
  return mapping::getBlockSize() / mapping::getWarpSize();
}

#pragma omp end declare variant

///}

#pragma omp end declare target
//} // namespace

#pragma omp declare target

bool mapping::isMainThreadInGenericMode() {
  if (mapping::isSPMDMode())
    return false;

  // Check if this is the last warp in the block.
  return mapping::getWarpId() + 1 == mapping::getNumberOfWarpsInBlock();
}

bool mapping::isLeaderInWarp() {
  __kmpc_impl_lanemask_t Active = mapping::activemask();
  __kmpc_impl_lanemask_t LaneMaskLT = mapping::lanemaskLT();
  return utils::popc(Active & LaneMaskLT) == 0;
}

LaneMaskTy mapping::activemask() { return activemaskImpl(); }
LaneMaskTy mapping::lanemaskLT() { return lanemaskLTImpl(); }
LaneMaskTy mapping::lanemaskGT() { return lanemaskGTImpl(); }
uint32_t mapping::getThreadIdInWarp() { return getThreadIdInWarpImpl(); }
uint32_t mapping::getThreadIdInBlock() { return getThreadIdInBlockImpl(); }
uint32_t mapping::getBlockSize() {
  return getBlockSizeImpl();
}
uint32_t mapping::getKernelSize() {
  return getKernelSizeImpl();
}
uint32_t mapping::getBlockId() { return getBlockIdImpl(); }
uint32_t mapping::getNumberOfBlocks() { return getNumberOfBlocksImpl(); }
uint32_t mapping::getNumberOfProcessorElements() {
  return getNumberOfProcessorElementsImpl();
}
uint32_t mapping::getWarpId() { return getWarpIdImpl(); }
uint32_t mapping::getWarpSize() { return getWarpSizeImpl(); }
uint32_t mapping::getNumberOfWarpsInBlock() {
  return getNumberOfWarpsInBlockImpl();
}

/// TODO
static int SHARED(IsSPMDMode);

void mapping::init(bool IsSPMD) { IsSPMDMode = IsSPMD; }

bool mapping::isSPMDMode() { return IsSPMDMode; }

bool mapping::isGenericMode() { return !isSPMDMode(); }

#pragma omp end declare target
