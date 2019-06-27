//===--- target_impl.h - OpenMP device RTL target code impl. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of target specific functions needed in the generic part of the
// device RTL implementation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_IMPL_H
#define TARGET_IMPL_H

#include "omptarget-nvptx.h"

/// Atomically increment the pointee of \p Ptr by \p Val and return the original
/// value of the pointee.
template <typename T> T __kmpc_impl_atomic_add(T *Ptr, T Val) {
  return atomicAdd(Ptr, Val);
}

/// Atomically exchange the pointee of \p Ptr with \p Val and return the
/// original value of the pointee.
template <typename T> T __kmpc_impl_atomic_exchange(T *Ptr, T Val) {
  return atomicExch(Ptr, Val);
}

/// Return the bit-mask representing active threads.
template <typename T> T __kmpc_impl_active_thread_mask() {
  return __ACTIVEMASK();
}

/// Perform an "omp flush" operation.
void __kmpc_impl_flush(kmp_Ident *) {
  __threadfence_system();
}

/// Perform an "omp barrier" operation for various modes described as
/// combinations of "(non)-cancellable", "(non-)simple", and "(non-)SPMD".
///
/// Note: A team is a block: we can use CUDA native synchronization mechanism.
///
/// FIXME: What if not all threads (warps) participate to the barrier? We may
///        need to implement it differently
template <bool IsCancellable, bool IsSimple, bool IsSPMD>
__kmpc_impl_barrier(kmp_Ident *Loc, int32_t TID) {
  // Try to justify SPMD mode first as it allows a simple barrier
  // implementation.
  bool InSPMD = IsSPMD || checkRuntimeUninitialized(Loc) || checkSPMDMode(Loc);

  if (InSPMD) {
    PRINT(LD_SYNC, "call kmpc%s_barrier%s_spmd\n",
          IsCancellable ? "_cancel" : "", IsSimple ? "_simple" : "");
    // FIXME: use __syncthreads instead when the function copy is fixed in LLVM.
    __SYNCTHREADS();
  } else {
    int NumberOfActiveOMPThreads = GetNumberOfOmpThreads(InSPMD);
    if (NumberOfActiveOMPThreads > 1) {
      // The #threads parameter must be rounded up to the WARPSIZE.
      int NumThreads =
          WARPSIZE * ((NumberOfActiveOMPThreads + WARPSIZE - 1) / WARPSIZE);

      PRINT(LD_SYNC,
            "call kmpc%s_barrier%s with %d omp NumThreads, sync parameter %d\n",
            IsCancellable ? "_cancel" : "", IsSimple ? "_simple" : "",
            NumberOfActiveOMPThreads, NumThreads);

      // Barrier #1 is for synchronization among active NumThreads.
      named_sync(L1_BARRIER, NumThreads);
    }
  }
  PRINT(LD_SYNC, "completed kmpc%s_barrier%s%s\n",
        IsCancellable ? "_cancel" : "", IsSimple ? "_simple" : "",
        InSPMD ? "_spmd" : "");
}

#endif // TARGET_IMPL_H
