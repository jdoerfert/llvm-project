//===--- sync.cpp --- OpenMP synchronization operations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic implementation for synchronization primitives.
//
//===----------------------------------------------------------------------===//

#include "debug.h"
#include "target_impl.h"

/// Perform a barrier operation that might cause a cancellation.
EXTERN int32_t __kmpc_cancel_barrier(kmp_Ident *Loc, int32_t TID) {
  __kmpc_impl_barrier</* IsCancellable */ true, /* IsSimple */ false,
                      /* IsSPMD */ false>(Loc, TID);
  return /* should be cancelled */ false;
}

/// Perform a barrier operation.
EXTERN void __kmpc_barrier(kmp_Ident *Loc, int32_t TID) {
  __kmpc_impl_barrier</* IsCancellable */ false, /* IsSimple */ false,
                      /* IsSPMD */ false>(Loc, TID);
}

/// Perform a simple barrier operation in SPMD-mode.
EXTERN void __kmpc_barrier_simple_spmd(kmp_Ident *Loc, int32_t TID) {
  __kmpc_impl_barrier</* IsCancellable */ false, /* IsSimple */ true,
                      /* IsSPMD */ true>(Loc, TID);
}

/// Perform a simple barrier operation in non-SPMD-mode.
EXTERN void __kmpc_barrier_simple_generic(kmp_Ident *Loc, int32_t TID) {
  __kmpc_impl_barrier</* IsCancellable */ false, /* IsSimple */ true,
                      /* IsSPMD */ false>(Loc, TID);
}

/// Function to be called at the beginning of an "ordered" region.
EXTERN void __kmpc_ordered(kmp_Ident *, int32_t) {
  PRINT0(LD_IO, "call kmpc_ordered\n");
}

/// Function to be called at the end of an "ordered" region.
EXTERN void __kmpc_end_ordered(kmp_Ident *, int32_t) {
  PRINT0(LD_IO, "call kmpc_end_ordered\n");
}

/// Create two functions, one to be called before entering region which returns
/// a non-zero value if the region should be entered, and one to be called after
/// the region was executed. The names of the function will be __kmpc_NAME and
/// __kmcp_end_NAME. The predicate under which the region is entered is provided
/// as ENTERING_PREDICATE.
#define REGION_DELIMITERS(NAME, ENTERING_PREDICATE)                            \
                                                                               \
  EXTERN int32_t __kmpc_##NAME(kmp_Ident *, int32_t GlobalTID) {               \
    PRINT0(LD_IO, "call " #NAME "\n");                                         \
    return ENTERING_PREDICATE(GlobalTID);                                      \
  }                                                                            \
                                                                               \
  EXTERN void __kmpc_end_##NAME(kmp_Ident *, int32_t GlobalTID) {              \
    PRINT0(LD_IO, "call " #NAME "\n");                                         \
    ASSERT0(LT_FUSSY, ENTERING_PREDICATE(GlobalTID),                           \
            "Region end function executed by thread which should not have "    \
            "entered");                                                        \
  }

/// Region delimiter functions for "master".
///{
REGION_DELIMITERS(master, IsTeamMaster)
///}

/// Region delimiter functions for "single" implemented the same as master.
///{
REGION_DELIMITERS(single, IsTeamMaster)
///}

/// Perform a "flush" operation.
EXTERN void __kmpc_flush(kmp_Ident *Loc) {
  PRINT0(LD_IO, "call kmpc_flush\n");
  __kmpc_impl_flush(Loc);
}

/// Return the bit-mask of active threads in the warp.
///
/// FIXME: Warps are a detail we should get rid of here.
EXTERN int32_t __kmpc_warp_active_thread_mask() {
  PRINT0(LD_IO, "call __kmpc_warp_active_thread_mask\n");
  return __kmpc_impl_active_thread_mask();
}
