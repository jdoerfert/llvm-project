; RUN: opt < %s -S -ipsccp | FileCheck %s
; RUN: opt -S -passes=attributor -aa-pipeline='basic-aa' -attributor-disable=false -attributor-max-iterations-verify -attributor-max-iterations=3 < %s | FileCheck %s --check-prefix=ATTRIBUTOR
; PR5596

; IPSCCP should propagate the 0 argument, eliminate the switch, and propagate
; the result.

; CHECK: define i32 @main() #0 {
; CHECK-NEXT: entry:
; CHECK-NOT: call
; CHECK-NEXT: ret i32 123

; FIXME: Remove obsolete calls/instructions
; ATTRIBUTOR: define i32 @main() #0 {
; ATTRIBUTOR-NEXT: entry:
; ATTRIBUTOR-NEXT: call
; ATTRIBUTOR-NEXT: ret i32 123

define i32 @main() noreturn nounwind {
entry:
  %call2 = tail call i32 @wwrite(i64 0) nounwind
  ret i32 %call2
}

define internal i32 @wwrite(i64 %i) nounwind readnone {
entry:
  switch i64 %i, label %sw.default [
    i64 3, label %return
    i64 10, label %return
  ]

sw.default:
  ret i32 123

return:
  ret i32 0
}

; CHECK: attributes #0 = { noreturn nounwind }
; CHECK: attributes #1 = { nounwind readnone }
