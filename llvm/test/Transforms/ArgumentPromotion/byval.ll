; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

%struct.ss = type { i32, i64 }

define internal void @f(%struct.ss* byval  %b) nounwind  {
entry:
; CHECK: define internal void @f(i32 %[[B0:[a-zA-Z0-9._-]*]], i64 %[[B1:[a-zA-Z0-9._-]*]])
; CHECK: alloca %struct.ss{{$}}
; CHECK: store i32 %[[B0]]
; CHECK: store i64 %[[B1]]
  %tmp = getelementptr %struct.ss, %struct.ss* %b, i32 0, i32 0
  %tmp1 = load i32, i32* %tmp, align 4
  %tmp2 = add i32 %tmp1, 1
  store i32 %tmp2, i32* %tmp, align 4
  ret void
}

define internal void @g(%struct.ss* byval align 32 %b) nounwind {
entry:
; CHECK: define internal void @g(i32 %[[B0:[a-zA-Z0-9._-]*]], i64 %[[B1:[a-zA-Z0-9._-]*]])
; CHECK: alloca %struct.ss, align 32
; CHECK: store i32 %[[B0]]
; CHECK: store i64 %[[B1]]
  %tmp = getelementptr %struct.ss, %struct.ss* %b, i32 0, i32 0
  %tmp1 = load i32, i32* %tmp, align 4
  %tmp2 = add i32 %tmp1, 1
  store i32 %tmp2, i32* %tmp, align 4
  ret void
}

define internal void @h([2 x i32]* byval %b) nounwind {
entry:
; Even if we do not access the first element we can promote this array.
  %tmp = getelementptr [2 x i32], [2 x i32]* %b, i32 0, i32 1
  %tmp1 = load i32, i32* %tmp, align 4
  %tmp2 = add i32 %tmp1, 1
  store i32 %tmp2, i32* %tmp, align 4
  ret void
}

define i32 @main() nounwind  {
entry:
; CHECK-LABEL: define i32 @main
  %A = alloca [2 x i32]
  %S = alloca %struct.ss
  %tmp1 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 0
  store i32 1, i32* %tmp1, align 8
  %tmp4 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 1
  store i64 2, i64* %tmp4, align 4

  call void @f(%struct.ss* byval %S) nounwind
; CHECK: call void @f(i32 %{{.*}}, i64 %{{.*}})

  call void @g(%struct.ss* byval %S) nounwind
; CHECK: call void @g(i32 %{{.*}}, i64 %{{.*}})

; Verify we unpack the byval array.
; FIXME: this sould be: call void @h(i32 %{{[a-zA-Z._0-9]*}}, i32 %{{[a-zA-Z._0-9]*}})
; CHECK: call void @h([2 x i32]* byval %A)
  call void @h([2 x i32]* byval %A) nounwind
  ret i32 0
}
