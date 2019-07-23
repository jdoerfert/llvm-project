; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s  -aa-pipeline='basic-aa' -passes=argpromotion -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.S = type { i32, i32 }

; Make sure we do not assume @unknown() is going to return which would
; allow us to pre-load %s in @caller. (PR42039)
define i32 @caller(%struct.S* %s) {
entry:
; CHECK: %call = call i32 @local(%struct.S* %s)
  %call = call i32 @local(%struct.S* %s)
  ret i32 %call
}

define internal i32 @local(%struct.S* noalias %s) {
entry:
  call void @unknown()
  %a = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  %0 = load i32, i32* %a, align 4
  %b = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

declare void @unknown() nounwind


; Make sure we do assume @unknown_willreturn() is going to return
; which allows us to pre-load %s in @caller_willreturn.
define i32 @caller_willreturn(%struct.S* %s) {
entry:
; CHECK: %call = call i32 @local_willreturn(i32 %{{.*}}, i32 %{{.*}})
  %call = call i32 @local_willreturn(%struct.S* %s)
  ret i32 %call
}

define internal i32 @local_willreturn(%struct.S* noalias %s) {
entry:
  call void @unknown_willreturn()
  %g0 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  %l0 = load i32, i32* %g0, align 4
  %g1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1
  %l1 = load i32, i32* %g1, align 4
  %add = add nsw i32 %l0, %l1
  ret i32 %add
}

declare void @unknown_willreturn() willreturn nounwind


; We should promote %s as it is for sure executed in
; @caller_must_be_executed but not %p as it is not always accessed (completely).
define i32 @caller_must_be_executed(%struct.S* %s, %struct.S* %p, i1 %c) {
entry:
; TODO: %call = call i32 @local_must_be_executed(i32 %{{.*}}, i32 %{{.*}}, %struct.S* %p, i1 %c)
  %call = call i32 @local_must_be_executed(%struct.S* %s, %struct.S* %p, i1 %c)
  ret i32 %call
}

define internal i32 @local_must_be_executed(%struct.S* %s, %struct.S* %p, i1 %c) {
entry:
  %sg0 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  %sv0 = load i32, i32* %sg0, align 4
  br i1 %c, label %then, label %else

then:
  %pg0 = getelementptr inbounds %struct.S, %struct.S* %p, i64 0, i32 0
  %pv0 = load i32, i32* %pg0, align 4
  %add0 = add nsw i32 %sv0, %pv0
  br label %merge

else:
  %pg1 = getelementptr inbounds %struct.S, %struct.S* %p, i64 0, i32 1
  %pv1 = load i32, i32* %pg1, align 4
  %add1 = add nsw i32 %sv0, %pv1
  br label %merge

merge:
  %phi = phi i32 [%add0, %then], [%add1, %else]
  %sg1 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 1
  %sv1 = load i32, i32* %sg1, align 4
  %add = add nsw i32 %phi, %sv1
  ret i32 %add
}
