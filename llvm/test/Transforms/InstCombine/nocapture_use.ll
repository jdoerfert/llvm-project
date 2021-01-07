; RUN: opt -instcombine -S < %s | FileCheck %s

%struct.S = type { i32* }

define i32 @base_negative() {
; The base case, we cannot propagate 1 to the return.
; CHECK: @base_negative
; CHECK: ret i32 %tmp2

  %local = alloca i32
  %s = alloca %struct.S
  %tmp = bitcast i32* %local to i8*
  %ptr = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  store i32* %local, i32** %ptr
  %call = call i32* @unknown_nocapture_pointer_in_arg(%struct.S* nonnull %s)
  store i32 1, i32* %local
  store i32 2, i32* %call
  %tmp2 = load i32, i32* %local
  ret i32 %tmp2
}

define i32 @nocapture_late_positive() {
; The nocapture case, store late, we can propagate 1 to the return.
; CHECK: @nocapture_late_positive
; CHECK: ret i32 1

  %local = alloca i32
  %s = alloca %struct.S
  %tmp = bitcast i32* %local to i8*
  %ptr = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  store i32* %local, i32** %ptr, !nocapture !0
  %call = call i32* @unknown_nocapture_pointer_in_arg(%struct.S* nonnull %s) ["nocapture_use"(i32* %local)]
  store i32 1, i32* %local
  store i32 2, i32* %call
  %tmp2 = load i32, i32* %local
  ret i32 %tmp2
}

define i32 @nocapture_early_negative() {
; The nocapture case, store early, we cannot propagate 1 to the return.
;
; The purpose of this test is to confirm that the use in
; @unknown_nocapture_pointer_in_arg can still overwrite
; the value stored in %local even if the pointer is not captured.
;
; CHECK: @nocapture_early_negative
; CHECK: ret i32 %tmp2

  %local = alloca i32
  %s = alloca %struct.S
  %tmp = bitcast i32* %local to i8*
  %ptr = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  store i32* %local, i32** %ptr, !nocapture !0
  store i32 1, i32* %local
  %call = call i32* @unknown_nocapture_pointer_in_arg(%struct.S* nonnull %s) ["nocapture_use"(i32* %local)]
  store i32 2, i32* %call
  %tmp2 = load i32, i32* %local
  ret i32 %tmp2
}

define i32 @nocapture_early_positive_no_operand_bundle() {
; The nocapture case, store early, we can propagate 1 to the return.
;
; The purpose of this test is to confirm that it is the operand bundle
; use in the call to @unknown_nocapture_pointer_in_arg which can still
; overwrite the value stored in %local even if the pointer is not captured.
;
; CHECK: @nocapture_early_positive_no_operand_bundle
; CHECK: ret i32 1

  %local = alloca i32
  %s = alloca %struct.S
  %tmp = bitcast i32* %local to i8*
  %ptr = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  store i32* %local, i32** %ptr, !nocapture !0
  store i32 1, i32* %local
  %call = call i32* @unknown_nocapture_pointer_in_arg(%struct.S* nonnull %s)
  store i32 2, i32* %call
  %tmp2 = load i32, i32* %local
  ret i32 %tmp2
}

declare dso_local i32* @unknown_nocapture_pointer_in_arg(%struct.S*)

!0 = !{}
