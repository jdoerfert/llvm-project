; RUN: opt -basicaa -print-all-alias-modref-info -aa-eval -analyze < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@G = global i32 0, align 4

define i16 @global_and_maxobj_arg_1(i16* maxobjsize(2) %arg) {
; CHECK:     Function: global_and_maxobj_arg_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* @G, i16* %arg
bb:
  store i16 1, i16* %arg, align 8
  store i32 0, i32* @G, align 4
  %tmp = load i16, i16* %arg, align 8
  ret i16 %tmp
}

define i8 @global_and_maxobj_arg_2(i8* maxobjsize(3) %arg) {
; CHECK:     Function: global_and_maxobj_arg_2: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i8* %arg, i32* @G
bb:
  store i8 1, i8* %arg, align 8
  store i32 0, i32* @G, align 4
  %tmp = load i8, i8* %arg, align 8
  ret i8 %tmp
}

define i8 @deref_and_maxobj_arg_1(i8* maxobjsize(6) %maxobj, i8* dereferenceable(7) %deref) {
; CHECK:     Function: deref_and_maxobj_arg_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i8* %maxobj, i8* %deref
bb:
  store i8 1, i8* %maxobj, align 8
  store i8 0, i8* %deref, align 8
  %tmp = load i8, i8* %maxobj, align 8
  ret i8 %tmp
}

declare maxobjsize(4) i32* @get_i32_maxobj4()
declare maxobjsize(4) i64* @get_i64_maxobj4()
declare void @unknown32(i32*)
declare void @unknown64(i64*)

define i64 @local_and_maxobj_ret_1() {
; CHECK:     Function: local_and_maxobj_ret_1: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i64* %obj, i64* %ret
bb:
  %obj = alloca i64
  call void @unknown64(i64* %obj)
  %ret = call maxobjsize(4) i64* @get_i64_maxobj4()
  store i64 1, i64* %obj, align 4
  %bc = bitcast i64* %ret to i32*
  store i32 0, i32* %bc, align 8
  %tmp = load i64, i64* %obj, align 4
  ret i64 %tmp
}

define i64 @local_and_maxobj_ret_2() {
; CHECK:     Function: local_and_maxobj_ret_2: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i64* %obj, i32* %ret
bb:
  %obj = alloca i64
  call void @unknown64(i64* %obj)
  %ret = call maxobjsize(4) i32* @get_i32_maxobj4()
  store i64 1, i64* %obj, align 4
  store i32 0, i32* %ret, align 8
  %tmp = load i64, i64* %obj, align 4
  ret i64 %tmp
}


; Baseline tests, same as above but with >=4 instead of <4 maxobjsize bytes.

define i16 @global_and_maxobj_arg_1b(i16* maxobjsize(4) %arg) {
; CHECK:     Function: global_and_maxobj_arg_1: 2 pointers, 0 call sites
; CHECK-NEXT:  MayAlias:	i32* @G, i16* %arg
bb:
  store i16 1, i16* %arg, align 8
  store i32 0, i32* @G, align 4
  %tmp = load i16, i16* %arg, align 8
  ret i16 %tmp
}

define i8 @global_and_maxobj_arg_2b(i8* maxobjsize(6) %arg) {
; CHECK:     Function: global_and_maxobj_arg_2: 2 pointers, 0 call sites
; CHECK-NEXT:  MayAlias:	i8* %arg, i32* @G
bb:
  store i8 1, i8* %arg, align 8
  store i32 0, i32* @G, align 4
  %tmp = load i8, i8* %arg, align 8
  ret i8 %tmp
}

declare maxobjsize(8) i32* @get_i32_maxobj8()
declare maxobjsize(8) i64* @get_i64_maxobj8()

define i32 @local_and_maxobj_ret_non_maxobj_1() {
; CHECK:     Function: local_and_maxobj_ret_non_maxobj_1: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i64* %ret
bb:
  %obj = alloca i32
  call void @unknown32(i32* %obj)
  %ret = call maxobjsize(8) i64* @get_i64_maxobj8()
  store i32 1, i32* %obj, align 4
  store i64 0, i64* %ret, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}

define i32 @local_and_maxobj_ret_non_maxobj_2() {
; CHECK:     Function: local_and_maxobj_ret_non_maxobj_2: 2 pointers, 2 call sites
; Different result than above (see @local_and_maxobj_ret_2).
; CHECK-NEXT:  MayAlias:	i32* %obj, i32* %ret
bb:
  %obj = alloca i32
  call void @unknown32(i32* %obj)
  %ret = call maxobjsize(8) i32* @get_i32_maxobj8()
  store i32 1, i32* %obj, align 4
  store i32 0, i32* %ret, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}
