; RUN: opt < %s -attributor-disable=false -attributor -functionattrs -S | FileCheck %s
; RUN: opt < %s -attributor-disable=false -passes="attributor,cgscc(function-attrs)" -S | FileCheck %s

; FIXME: Because nocapture and readnone deduction in the functionattrs pass are interleaved, it doesn't
;        trigger when nocapture is already present. Once the Attributor derives memory behavior,
;        this should be fixed.
; FIXME: readnone missing for %0 two times
; CHECK: define void @bar(i8* nocapture readonly)
define void @bar(i8* readonly %0) {
  call void @foo(i8* %0)
    ret void
}

; CHECK: define void @foo(i8* nocapture readonly)
define void @foo(i8* readonly %0) {
  call void @bar(i8* %0)
  ret void
}
