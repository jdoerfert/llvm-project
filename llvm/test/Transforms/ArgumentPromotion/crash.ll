; RUN: opt -S < %s -inline -argpromotion | FileCheck %s
; RUN: opt -S < %s -passes=inline,argpromotion | FileCheck %s

%S = type { %S* }

; Inlining should nuke the invoke (and any inlined calls) here even with
; argument promotion running along with it.
define void @zot() personality i32 (...)* @wibble {
; CHECK-LABEL: define void @zot() personality i32 (...)* @wibble
; CHECK-NOT: call
; CHECK-NOT: invoke
bb:
  invoke void @hoge()
          to label %bb1 unwind label %bb2

bb1:
  unreachable

bb2:
  %tmp = landingpad { i8*, i32 }
          cleanup
  unreachable
}

define internal void @hoge() {
bb:
  %tmp = call fastcc i8* @spam(i1 (i8*)* @eggs)
  %tmp1 = call fastcc i8* @spam(i1 (i8*)* @barney)
  unreachable
}

define internal fastcc i8* @spam(i1 (i8*)* %arg) {
bb:
  unreachable
}

define internal i1 @eggs(i8* %arg) {
bb:
  %tmp = call zeroext i1 @barney(i8* %arg)
  unreachable
}

define internal i1 @barney(i8* %arg) {
bb:
  ret i1 undef
}

define i32 @test_inf_promote_caller(i32 %arg) {
; CHECK-LABEL: define i32 @test_inf_promote_caller(
bb:
  %tmp = alloca %S
  %tmp1 = alloca %S
  %tmp2 = call i32 @test_inf_promote_callee(%S* %tmp, %S* %tmp1)
; CHECK: call i32 @test_inf_promote_callee(%S* %{{.*}}, %S* %{{.*}})

  ret i32 0
}

define internal i32 @test_inf_promote_callee(%S* %arg, %S* %arg1) {
; CHECK-LABEL: define internal i32 @test_inf_promote_callee(
; CHECK: %S* %{{.*}}, %S* %{{.*}})
bb:
  %tmp = getelementptr %S, %S* %arg1, i32 0, i32 0
  %tmp2 = load %S*, %S** %tmp
  %tmp3 = getelementptr %S, %S* %arg, i32 0, i32 0
  %tmp4 = load %S*, %S** %tmp3
  %tmp5 = call i32 @test_inf_promote_callee(%S* %tmp4, %S* %tmp2)
; CHECK: call i32 @test_inf_promote_callee(%S* %{{.*}}, %S* %{{.*}})

  ret i32 0
}

define i32 @test_inf2_promote_caller(i32 %arg) {
; CHECK-LABEL: define i32 @test_inf2_promote_caller(
bb:
  %tmp = alloca %S
  %tmp1 = alloca %S
  %tmp2 = call i32 @test_inf2_promote_callee(%S* %tmp, %S* %tmp1)
; CHECK: call i32 @test_inf2_promote_callee(%S* %{{.*}}, %S* %{{.*}})

  ret i32 0
}

; Recursion but not self-recursion
define internal i32 @test_inf2_promote_passthrough(%S* %arg, %S* %arg1) noinline {
  %tmp0 = call i32 @test_inf2_promote_callee(%S* %arg, %S* %arg1)
  ret i32 0
}

define internal i32 @test_inf2_promote_callee(%S* %arg, %S* %arg1) noinline {
; CHECK-LABEL: define internal i32 @test_inf2_promote_callee(
; CHECK: %S* %{{.*}}, %S* %{{.*}})
bb:
  %tmp = getelementptr %S, %S* %arg1, i32 0, i32 0
  %tmp2 = load %S*, %S** %tmp
  %tmp3 = getelementptr %S, %S* %arg, i32 0, i32 0
  %tmp4 = load %S*, %S** %tmp3
; PR42683
  %tmp5 = call i32 @test_inf2_promote_passthrough(%S* %tmp4, %S* %tmp2)
; CHECK: call i32 @test_inf2_promote_passthrough(%S* %{{.*}}, %S* %{{.*}})
  ret i32 0
}

declare i32 @wibble(...)

%type_opaque = type opaque

; CHECK: define internal fastcc i32 @f0(i32)
; CHECK: ret i32 %0
define internal fastcc i32 @f0(%type_opaque* nocapture readonly %a0) noinline {
entry:
  %0 = bitcast %type_opaque* %a0 to i8*
  %add.ptr = getelementptr i8, i8* %0, i32 24
  %1 = bitcast i8* %add.ptr to i32*
  %2 = load i32, i32* %1, align 16
  ret i32 %2
}

; CHECK:      define fastcc i32 @f1(%type_opaque* nocapture readonly %a0)
; CHECK-NEXT:   %[[ptr:[^ ]*]] = bitcast %type_opaque* %a0 to i8*
; CHECK-NEXT:   %[[gep:[^ ]*]] = getelementptr i8, i8* %[[ptr]], i32 24
; CHECK-NEXT:   %[[cst:[^ ]*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT:   %[[val:[^ ]*]] = load i32, i32* %[[cst]], align 16
; CHECK-NEXT:   %t = call fastcc i32 @f0(i32 %[[val]])
; CHECK-NEXT:   ret i32 %t

define fastcc i32 @f1(%type_opaque* nocapture readonly %a0) {
  %t = call fastcc i32 @f0(%type_opaque* %a0)
  ret i32 %t
}
