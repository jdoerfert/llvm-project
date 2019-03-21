; RUN: opt < %s -attributor -attributor-disable=false -functionattrs -S | FileCheck %s
; RUN: opt < %s -passes='attributor,cgscc(function-attrs)' -attributor-disable=false -S | FileCheck %s
; PR8279

@g = constant i32 1

; CHECK: Function Attrs
; CHECK-SAME: norecurse
; CHECK-NOT: readonly
; CHECK-NEXT: void @foo()
define void @foo() {
  %tmp = load volatile i32, i32* @g
  ret void
}
