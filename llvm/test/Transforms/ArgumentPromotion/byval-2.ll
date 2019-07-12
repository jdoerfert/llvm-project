; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

; Arg promotion eliminates the struct argument.

%struct.ss = type { i32, i64 }

define internal void @f(%struct.ss* byval  %b, i32* byval %X) nounwind  {
; CHECK-LABEL: define internal void @f(i32 %{{.*}}, i64 %{{.*}}, i32 %{{.*}})
entry:
  %tmp = getelementptr %struct.ss, %struct.ss* %b, i32 0, i32 0
  %tmp1 = load i32, i32* %tmp, align 4
  %tmp2 = add i32 %tmp1, 1
  store i32 %tmp2, i32* %tmp, align 4

  store i32 0, i32* %X
  ret void
}

define i32 @test(i32* %X) {
; CHECK-LABEL: define i32 @test
entry:
  %S = alloca %struct.ss
  %tmp1 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 0
  store i32 1, i32* %tmp1, align 8
  %tmp4 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 1
  store i64 2, i64* %tmp4, align 4
  call void @f( %struct.ss* byval %S, i32* byval %X)
; CHECK: call void @f(i32 %{{.*}}, i64 %{{.*}}, i32 %{{.*}})
  ret i32 0
}
