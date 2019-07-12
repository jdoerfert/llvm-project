; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

%struct.ss = type { i32, i64 }

define internal i32 @f(%struct.ss* byval %b, i32* byval %X, i32 %i) nounwind {
; CHECK-LABEL: define internal i32 @f(
; CHECK: i32 %[[B0:.*]], i64 %[[B1:.*]], i32 %[[XV:.*]], i32 %i)
entry:
; CHECK: %[[B:.*]] = alloca %struct.ss
; CHECK: %[[B_GEP0:.*]] = bitcast %struct.ss* %[[B]] to i32*
; CHECK: store i32 %[[B0]], i32* %[[B_GEP0]]
; CHECK: %[[B_GEP1:.*]] = getelementptr %struct.ss, %struct.ss* %[[B]], i32 0, i32 1
; CHECK: store i64 %[[B1]], i64* %[[B_GEP1]]
; CHECK: %[[X:.*]] = alloca i32
; CHECK: store i32 %[[XV]], i32* %[[X]]

  %tmp = getelementptr %struct.ss, %struct.ss* %b, i32 0, i32 0
; CHECK: %[[TMP:.*]] = getelementptr %struct.ss, %struct.ss* %[[B]], i32 0, i32 0
  %tmp1 = load i32, i32* %tmp, align 4
; CHECK: %[[TMP1:.*]] = load i32, i32* %[[TMP]]
  %tmp2 = add i32 %tmp1, 1
; CHECK: %[[TMP2:.*]] = add i32 %[[TMP1]], 1
  store i32 %tmp2, i32* %tmp, align 4
; CHECK: store i32 %[[TMP2]], i32* %[[TMP]]

  store i32 0, i32* %X
; CHECK: store i32 0, i32* %X
  ret i32 %i
}

; Make sure we don't drop the call zeroext attribute.
define i32 @test(i32* %X) {
; CHECK-LABEL: define i32 @test(
entry:
  %S = alloca %struct.ss
; CHECK: %[[S:.*]] = alloca %struct.ss
  %tmp1 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 0
  store i32 1, i32* %tmp1, align 8
; CHECK: store i32 1
  %tmp4 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 1
  store i64 2, i64* %tmp4, align 4
; CHECK: store i64 2

  call i32 @f( %struct.ss* byval %S, i32* byval %X, i32 zeroext 0)
; CHECK: %[[S_GEP0:.*]] = getelementptr %struct.ss, %struct.ss* %[[S]], i32 0, i32 0
; CHECK: %[[S0:.*]] = load i32, i32* %[[S_GEP0]]
; CHECK: %[[S_GEP1:.*]] = getelementptr %struct.ss, %struct.ss* %[[S]], i32 0, i32 1
; CHECK: %[[S1:.*]] = load i64, i64* %[[S_GEP1]]
; CHECK: %[[XVal:.*]] = load i32, i32* %X
; CHECK: call i32 @f(i32 %[[S0]], i64 %[[S1]], i32 %[[XVal]], i32 zeroext 0)

  ret i32 0
}
