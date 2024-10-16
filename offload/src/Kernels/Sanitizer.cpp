//===-- Kenrels/Sanitizer.cpp - Sanitizer Kernel Definitions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cstdint>

extern "C" {
__device__ void __offload_san_register_host(void *Ptr, uint64_t Size,
                                            uint32_t SlotId);
__device__ void __offload_san_unregister_host(void *Ptr);
__device__ void __offload_san_get_ptr_info(uint32_t SlotId, void **Ptr,
                                           uint64_t *Size,
                                           uint64_t *LocationIdPtr);
__device__ void __offload_san_get_global_info(void *Ptr,
                                              uint64_t *LocationIdPtr);

[[gnu::weak, clang::disable_sanitizer_instrumentation]] __global__ void
__sanitizer_register(void *P, uint64_t Bytes, uint64_t SlotId) {
  __offload_san_register_host(P, Bytes, SlotId);
}

[[gnu::weak, clang::disable_sanitizer_instrumentation]] __global__ void
__sanitizer_unregister(void *P) {
  __offload_san_unregister_host(P);
}

[[gnu::weak, clang::disable_sanitizer_instrumentation]] __global__ void
__sanitizer_get_ptr_info(uint32_t SlotId, void **Ptr, uint64_t *Size,
                         uint64_t *LocationIdPtr) {
  __offload_san_get_ptr_info(SlotId, Ptr, Size, LocationIdPtr);
}

[[gnu::weak, clang::disable_sanitizer_instrumentation]] __global__ void
__sanitizer_get_global_info(void *Ptr, uint64_t *LocationIdPtr) {
  __offload_san_get_global_info(Ptr, LocationIdPtr);
}
}
