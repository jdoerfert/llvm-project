; RUN: opt -S --input-gen-mode=generate --passes=input-gen-instrument-entries,input-gen-instrument-memory %s | FileCheck %s

; Check that we generate a stub for decl_i32:

; CHECK-LABEL: define private i32 @__ig_decl_i32() {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = alloca i32, align 4
; CHECK-NEXT:    call void @__ig_gen_value(ptr [[TMP0]], i32 4, i64 4, i32 12)
; CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4
; CHECK-NEXT:    ret i32 [[TMP1]]

; Check that we also generate the gen_value sequence at the callsite:

; CHECK-LABEL: define private i32 @__ig_foo()
; CHECK-NEXT:    [[TMP1:%.*]] = alloca i32, align 4
; CHECK-NEXT:    call void @__ig_gen_value(ptr [[TMP1]], i32 4, i64 4, i32 12)
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[TMP1]], align 4
; CHECK-NEXT:    ret i32 [[TMP2]]

declare i32 @decl_i32()

define i32 @foo() #0 {
  %v = call i32 () @decl_i32()
  ret i32 %v
}

attributes #0 = { inputgen_entry }
