//===- objsan/include/objsan_interface_internal.h -------------------------===//
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

#ifndef OBJSAN_INCLUDE_OBJSAN_INTERFACE_INTERNAL_H
#define OBJSAN_INCLUDE_OBJSAN_INTERFACE_INTERNAL_H

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

void __objsan_rt_init(void);
void __objsan_rt_deinit(void);

#ifdef __cplusplus
} // extern C
#endif

#pragma GCC visibility pop

#endif // OBJSAN_INCLUDE_OBJSAN_INTERFACE_INTERNAL_H
