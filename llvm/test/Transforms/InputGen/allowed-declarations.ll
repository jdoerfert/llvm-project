; RUN: opt -S \
; RUN:   --input-gen-allow-external-funcs=decl_1 \
; RUN:   --input-gen-allow-external-funcs=decl_2 \
; RUN:   --input-gen-mode=generate --passes=input-gen-instrument-entries,input-gen-instrument-memory %s | FileCheck %s

; CHECK: declare i32 @decl_1()
; CHECK: declare i32 @decl_2()

; CHECK-LABEL: define private i32 @__ig_foo(
; CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:    [[A:%.*]] = call i32 @decl_1()
; CHECK-NEXT:    [[B:%.*]] = call i32 @decl_2()
; CHECK-NEXT:    [[V:%.*]] = add i32 [[A]], [[B]]
; CHECK-NEXT:    ret i32 [[V]]

declare i32 @decl_1()
declare i32 @decl_2()

define i32 @foo() #0 {
  %a = call i32 () @decl_1()
  %b = call i32 () @decl_2()
  %c = add i32 %a, %b
  ret i32 %c
}

attributes #0 = { inputgen_entry }
